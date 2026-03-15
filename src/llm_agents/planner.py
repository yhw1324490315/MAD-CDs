# src/llm_agents/planner.py
#
# Comments cover: Module imports, environment loading, class and method design, behavior of each internal step, exception handling, log saving, tool calls and parsing, etc.

import os
import json
import re
import base64
from openai import OpenAI
from dotenv import load_dotenv
from src.llm_agents.data_tools import DataToolkit
from datetime import datetime
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client, log_to_global_file

# ----------------------------
# Environment & Configuration
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)


class PseudoToolCall:
    """
    Wrap the 'implicit tool call' structure extracted from the model with a lightweight object.

    This class does not actually perform a remote call, but converts the parsed JSON structure into 
    an object shaped like an openai tool call (containing id, function.name, function.arguments, etc.),
    to facilitate unified interaction with subsequent processing logic (e.g., messages list).
    """
    def __init__(self, data):
        # data is expected to be like {"id":"call_0","function": {"name":..., "arguments":...}}
        self.id = data['id']
        # Convert the function dictionary into a simple object for subsequent access via .name and .arguments
        # type('obj', (object,), data['function']) dynamically creates a class instance whose attributes come from dictionary keys
        self.function = type('obj', (object,), data['function'])
        self.type = 'function'


class PlannerAgent:
    """
    Responsibilities of PlannerAgent:
      - Receive user query (user_query)
      - Call the LLM to get the required tool calls (e.g., retrieve experimental data / query feature importance)
      - Execute these tools via DataToolkit, and inject the results into the dialogue history
      - Request the LLM to generate the final JSON report

    Design Notes:
      - Use get_llm_client() to encapsulate LLM client creation (compatible with getting different LLM clients)
      - Use ConfigLoader to manage prompts and other configurations
      - Maintain strict records of messages (dialogue history) for playback / logging
    """

    def __init__(self):
        # Load the configuration singleton (e.g., prompts, model settings, etc.)
        self.config_loader = ConfigLoader.get_instance()
        self.prompts = self.config_loader.prompts

        # Data toolkit: Encapsulates query capabilities for experimental databases/files (DataToolkit is implemented in other modules)
        self.toolkit = DataToolkit()

        # Open LLM client and model configuration (returns client, model_name, temperature)
        self.client, self.model, self.temperature = get_llm_client()
        
        # Declare the tool schema available to the LLM (used for function-calling or prompting the model on how to call tools)
        # Each tool contains type and function descriptions (name/description/parameters)
        # These details are used to let the LLM know which functions are callable, and the format/meaning of parameters
        self.tools_schema = [
             {
                "type": "function",
                "function": {
                    "name": "analyze_material_data",
                    "description": "Retrieve experimental data, which is crucial for understanding the relationship between precursors, processes, and performance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_type": {"type": "string", "enum": ["emission", "lifetime"]},
                            "min_value": {"type": "number"},
                            "max_value": {"type": "number"}
                        },
                        "required": ["target_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_feature_importance",
                    "description": "Obtain feature importance and SHAP analysis plots to identify key chemical groups (Bits) and process conditions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "enum": ["lifetime", "emission"]}
                        },
                        "required": ["target"]
                    }
                }
            }
        ]

    # --- Utility Functions ---
    def _encode_image(self, image_path):
        """
        Return the base64 encoded string of the image at the specified path.
        Return None to indicate the image does not exist or failed to be read.

        This function is used when a local image needs to be embedded in messages (e.g., sending a SHAP plot to the LLM).
        """
        if not os.path.exists(image_path): return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except: return None

    def _serialize_message(self, msg):
        """
        Convert a dialogue message into a basic Python type (dict/str, etc.) suitable for logging.
        Compatible with multiple message objects: if the message is a dict, return it directly;
        if it has to_dict() or model_dump() methods (e.g., pydantic objects), call them;
        otherwise, fall back to using str(msg).
        """
        if isinstance(msg, dict): return msg
        if hasattr(msg, 'to_dict'): return msg.to_dict()
        if hasattr(msg, 'model_dump'): return msg.model_dump()
        return str(msg)

    def _save_log(self, input_messages, raw_response, step):
        """
        Save an interaction with the LLM (input messages and raw response raw_response) to run_dir/logs.
        Contains:
        1. jsonl format (machine-readable)
        2. txt format (human-readable)
        """
        try:
            # Use unified run_dir (provided by the utility function get_run_dir)
            log_dir = os.path.join(get_run_dir(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # --- 1. JSONL Logging ---
            path = os.path.join(log_dir, "planner_interaction_log.jsonl")
            s_inputs = []
            for m in input_messages:
                m_d = self._serialize_message(m)
                # If the message content is a list containing images or multipart contents
                if isinstance(m_d.get('content'), list):
                    clean = []
                    for it in m_d['content']:
                        if it.get('type') == 'image_url': clean.append({"type":"image_url", "url":"[Base64 Truncated]"})
                        else: clean.append(it)
                    m_d = m_d.copy()
                    m_d['content'] = clean
                s_inputs.append(m_d)
            # Try to model_dump raw_response (if supported) or fall back to str
            resp_d = raw_response.model_dump() if hasattr(raw_response, 'model_dump') else str(raw_response)
            entry = {"dt":str(datetime.now()), "step":step, "in":s_inputs, "out":resp_d}
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

            # --- 2. Readable Global Logging ---
            # Format Input
            input_str = ""
            for m in input_messages:
                 # Handle both dict and SDK object types
                 if isinstance(m, dict):
                     role = m.get('role', 'unknown')
                     content = m.get('content', '')
                 else:
                     role = getattr(m, 'role', 'unknown')
                     content = getattr(m, 'content', '')
                 
                 if isinstance(content, list):
                     text_parts = [str(p.get('text', '')) for p in content if isinstance(p, dict) and p.get('type')=='text']
                     content = " ".join(text_parts) + " [Image content hidden]"
                 input_str += f"[{str(role).upper()}]:\n{str(content)}\n\n"
            
            # Format Output
            resp_content = raw_response.choices[0].message.content if hasattr(raw_response, 'choices') else str(raw_response)
            
            log_to_global_file("PlannerAgent", input_str, resp_content, f"Step {step}")

        except Exception as e:
            # Failure to save logs does not affect the main flow, but should give a visible prompt for troubleshooting
            print(f"⚠️ Log save failed: {e}")

    def _parse_tool_calls(self, text):
        """
        Parse 'tool call' snippets output by the LLM as free text.

        Scenario: Some LLMs do not use a structured tool_calls field, but output JSON lines or similar structures directly in the text.
        This function tries to parse the JSON (or JSON-like) objects line by line in the text and determines which tool should be called based on the content.

        Returns: A list, with items like {"id":"call_0", "function": {"name":..., "arguments": ...}}
        """
        found = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            try:
                # Some LLMs append a comma at the end of the line, remove it for parsing
                if line.endswith(','): line = line[:-1]
                data = json.loads(line)
                if not isinstance(data, dict): continue
                # If JSON contains the "name" field (possibly from tools schema), use it directly
                if "name" in data:
                    name = data["name"].replace("functions.", "")
                    args = json.dumps(data.get("arguments")) if isinstance(data.get("arguments"), dict) else data.get("arguments")
                    found.append({"id":f"call_{len(found)}", "function":{"name":name, "arguments":args}})
                # If JSON contains the target_type field, infer it as a call to analyze_material_data
                elif "target_type" in data:
                    found.append({"id":f"call_{len(found)}", "function":{"name":"analyze_material_data", "arguments":line}})
                # If JSON contains the target field, infer it as a call to query_feature_importance
                elif "target" in data:
                    found.append({"id":f"call_{len(found)}", "function":{"name":"query_feature_importance", "arguments":line}})
            except: continue
        return found

    def _clean_json(self, content):
        """
        Clean the JSON string from code block markdown labels or redundant text possibly generated by the LLM.

        Behavior:
          - If the text is wrapped by ``` ``` (possibly ```json), strip the wrappers
          - First try json.loads directly; if failed, use regex to grab the first curly brace object ({...}) as a fallback

        Returns: A string that can likely be parsed by json.loads; or a best-effort extracted snippet if it fails.
        """
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            json.loads(content)
            return content
        except:
            m = re.search(r"(\{.*\})", content, re.DOTALL)
            return m.group(1) if m else content

    def _print_tool_execution_details(self, func_name, args, result, image_path=None):
        """
        Print tool execution input, output, and statistics to the console in a readable manner, facilitating manual review.
        This is not a log file, but a real-time console-friendly visual output.
        """
        print("\n" + "="*80)
        print(f"🕵️  [Transparency Report] The agent is reviewing data")
        print(f"🛠️  Tool Name: {func_name}")
        print(f"📥  Input parameters: {json.dumps(args, ensure_ascii=False)}")
        print("-" * 80)
        
        if func_name == "analyze_material_data":
            count = result.get("count", 0)
            data_preview = result.get("data", [])
            print(f"📊  [Experimental Data] A total of {count} records retrieved.")
            print(f"    (Data Preview - Top 6 records):")
            print(json.dumps(data_preview[:6], ensure_ascii=False, indent=2))
            
        elif func_name == "query_feature_importance":
            # Display the Top Bits and Conditions returned from the feature importance analysis
            bits = (result.get("feature_importance_list_BITS_TOP_20_CSV") or 
                   result.get("candidate_bits_from_csv") or 
                   result.get("top_bits_from_csv", []))
            conds = (result.get("feature_importance_list_CONDITIONS_TOP_20_CSV") or 
                    result.get("candidate_conditions_from_csv") or 
                    result.get("top_conditions_from_csv", []))
            
            print(f"📋  [Feature List] Candidate Bits ({len(bits)}):")
            print(json.dumps(bits, ensure_ascii=False, indent=2))
            print(f"📋  [Crucial Conditions] Candidate Conditions ({len(conds)}):")
            print(json.dumps(conds, ensure_ascii=False, indent=2))
            
            if image_path:
                # If a SHAP image path is provided, inform the user and print the path
                print(f"🖼️  [Visual Input] Displaying SHAP plot: {image_path}")
            else:
                print("    (No corresponding SHAP plot found)")
                
        print("="*80 + "\n")

    def run(self, user_query):
        """
        Main execution entry point:
          1. Construct system + user messages
          2. Force call the two core tools (analyze_material_data + query_feature_importance)
          3. Append tool results back into messages
          4. Trigger LLM to generate the final JSON report based on the tool results

        Return values:
          - cleaned_json_str (str): The final JSON report generated by the LLM (cleaned to a JSON string)
          - execution_context (dict): Contains intermediate retrieved experimental data and feature importance summaries, facilitating further usage by external callers
        """
        # Read the planner_agent's system prompt from the config file
        system_prompt = get_prompt('planner_agent_system', '')
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        print(f"🤖 [Planner] Instruction received: {user_query}")

        # execution_context is used to return intermediate results externally, facilitating subsequent analysis or visualization
        execution_context = {
            "retrieved_experiments": [],    # Stores results of analyze_material_data
            "feature_importance_top5": {},  # Stores results of query_feature_importance
            "user_query": user_query
        }

        # ====================================================================
        # Force call the two core tools (No longer relying on LLM's tool_choice)
        # ====================================================================
        
        # 1. Determine target type (emission or lifetime)
        target_type = "emission"  # Default
        query_lower = user_query.lower()
        if "lifetime" in query_lower or "afterglow" in query_lower:
            target_type = "lifetime"
        if "emission" in query_lower or "wavelength" in query_lower:
            target_type = "emission"
        
        print(f"🎯 [Planner] Detected target type: {target_type}")
        
        # 2. Force call analyze_material_data
        print("\n" + "="*80)
        print("🔧 [Planner] Forcing tool call: analyze_material_data")
        material_result = self.toolkit.get_experiment_data_with_sampling(
            target_type=target_type,
            min_val=None,
            max_val=None
        )
        self._print_tool_execution_details("analyze_material_data", {"target_type": target_type}, material_result)
        
        if material_result.get("data"):
            execution_context["retrieved_experiments"].extend(material_result["data"])
        
        messages.append({
            "role": "assistant",
            "content": f"I have called the analyze_material_data tool to retrieve experimental data."
        })
        messages.append({
            "role": "user",
            "content": f"Tool return result (analyze_material_data):\n{json.dumps(material_result, ensure_ascii=False)}"
        })
        
        # 3. Force call query_feature_importance
        print("🔧 [Planner] Forcing tool call: query_feature_importance")
        # Ensure we get enough candidates (e.g., top 20) so the LLM can filter them down to 6-8 based on SHAP
        feature_result = self.toolkit.query_feature_importance(target_type, top_n=20)
        image_to_inject = feature_result.get('shap_image_path') if feature_result.get('shap_image_available') else None
        
        feature_summary = {
            "target": feature_result.get("target"),
            "feature_importance_list_BITS_TOP_20_CSV": feature_result.get("top_bits_from_csv"),
            "feature_importance_list_CONDITIONS_TOP_20_CSV": feature_result.get("top_conditions_from_csv"),
            "note": "SHAP image attached. Please use it to refine the selection." if image_to_inject else "No SHAP image."
        }
        self._print_tool_execution_details("query_feature_importance", {"target": target_type}, feature_summary, image_to_inject)
        
        execution_context["feature_importance_top5"] = {
            "top_bits": feature_result.get("top_bits_from_csv"),
            "top_conditions": feature_result.get("top_conditions_from_csv")
        }
        
        messages.append({
            "role": "user",
            "content": f"Tool return result (query_feature_importance) - Candidates:\n{json.dumps(feature_summary, ensure_ascii=False)}"
        })
        
        # 4. If a SHAP image exists, inject it into messages
        if image_to_inject:
            b64 = self._encode_image(image_to_inject)
            if b64:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"This is the SHAP summary plot for {target_type}. The plot shows the direction and magnitude of the impact of each feature on the target value."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                })
        
        # 5. Inform the LLM that data collection is complete and to generate the final report
        messages.append({
            "role": "user", 
            "content": """Data collection complete. 
            I have provided provided candidate lists of important features (Bits and Conditions) from the global analysis CSV and the SHAP summary plot.

            **CRITICAL TASK**:
            1. Analyze the SHAP plot to identify which features have the most distinct and consistent impact.
            2. Select the **Final 10 Most Critical Features** from the candidates.
            - **MANDATORY**: You MUST include the Top 3 Bits and Top 3 Conditions from the **"feature_importance_list_BITS_TOP_20_CSV"** and **"feature_importance_list_CONDITIONS_TOP_20_CSV"** provided above. Do not ignore them.
            - Combine these with insights from the SHAP plot.
            
            3. **Generate the final JSON report** with the following keys (ensure to include the new 'raw_feature_importance_top_csv' key):
            {
                "target_property": "...",
                "design_constraints": [...],
                "key_bits_to_decode": [ ... mixed selection ... ],
                "key_Importance": [ ... mixed selection ... ],
                "raw_feature_importance_top_csv": {
                    "bits": [ ... copy top 5 from CSV list ... ],
                    "conditions": [ ... copy top 5 from CSV list ... ]
                },
                "data_insights": "...",
                "task_type": "..."
            }"""
        })

        # ====================================================================
        # Request the LLM to generate the final JSON report based on the tool results
        # ====================================================================
        print("🤖 [Planner] Generating JSON...")
        try:
            final = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                response_format={"type": "json_object"},
                temperature=self.temperature
            )
            # Save interaction logs
            self._save_log(messages, final, 1)
            
            # Clean LLM output, extract JSON string
            cleaned_json_str = self._clean_json(final.choices[0].message.content)
            
            return cleaned_json_str, execution_context
            
        except Exception as e:
            print(f"❌ [Planner] LLM failed to generate report: {e}")
            # Return fallback JSON
            fallback_json = json.dumps({
                "target_property": target_type,
                "design_constraints": [],
                "key_bits_to_decode": execution_context["feature_importance_top5"].get("top_bits", [])[:8],
                "key_Importance": execution_context["feature_importance_top5"].get("top_conditions", [])[:5],
                "data_insights": f"Analyzed based on {len(execution_context['retrieved_experiments'])} experimental records",
                "error": str(e)
            }, ensure_ascii=False)
            return fallback_json, execution_context
