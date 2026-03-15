import os
import json
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from src.llm_agents.scout import ScoutAgent
from src.llm_agents.optimizer import OptimizerAgent
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client, log_to_global_file
from datetime import datetime

class ArchitectAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        
        # LLM Initialization
        self.client, self.model, self.temperature = get_llm_client()
        self.scout = ScoutAgent()
        self.optimizer = OptimizerAgent()

    def _save_readable_log(self, input_messages, raw_response, step_info="Architect"):
        """Save human-readable interaction log (global compilation)"""
        try:
             # Format Input
             input_str = ""
             for m in input_messages:
                 role = str(m.get('role', 'unknown'))
                 content = str(m.get('content', ''))
                 if len(content) > 10000:
                     content = content[:10000] + "... [TRUNCATED]"
                 input_str += f"[{role.upper()}]:\n{content}\n\n"
             
             # Format Output
             resp_content = raw_response.choices[0].message.content if raw_response.choices else ""
             
             log_to_global_file("ArchitectAgent", input_str, resp_content, step_info)
        except Exception as e:
             print(f"⚠️ Log save failed: {e}")

    def generate_recipe(self, summary_report_json, planner_context=None, scout_limit=100000, scout_max_mw=None, initial_query=None):
        """
        Generate experimental recipe in standard format
        """
        if planner_context is None:
            planner_context = {}
            
        print("\n🏗️ [Architect] Received intelligence, starting to draft Standard Operating Procedure (SOP)...")
        if initial_query:
            print(f"🎯 User original requirement: {initial_query}")

        # --- 1. Split & Execute ---
        
        try:
            molecule_candidates_raw = self.scout.search_molecules(
                summary_report_json, 
                limit=scout_limit,
                max_mw=scout_max_mw
            )
        except Exception as e:
            print(f"❌ [Architect] Scout execution failed: {e}")
            molecule_candidates_raw = []

        if not molecule_candidates_raw:
            print("⚠️ [Architect] No candidate molecules found, will attempt to use fallback strategy.")

        try:
            optimizer_result = self.optimizer.optimize(summary_report_json, molecule_candidates_raw)
        except Exception as e:
            print(f"❌ [Architect] Optimizer execution failed: {e}")
            optimizer_result = {}

        # --- 2. Merge ---
        
        molecule_candidates_optimized = optimizer_result.get("Molecules_With_Params", molecule_candidates_raw)
        process_params = {k:v for k,v in optimizer_result.items() if k != "Molecules_With_Params"}

        print("🔄 [Architect] Generating report according to standard template...")
        
        try:
            key_feature = summary_report_json.get('critical_features_analysis', [{}])[0].get('feature_name', 'Critical Chemical Group')
        except:
            key_feature = 'Target Functional Group'

        # Extract raw data from context
        raw_experiments = planner_context.get("retrieved_experiments", [])
        raw_features = planner_context.get("feature_importance_top5", {})
        
        # Load Template
        template = get_prompt('architect_agent_template')
        
        # 🚨 KEY: Inject user original design requirement into template
        user_goal_section = ""
        if initial_query:
            user_goal_section = f"""
### 🎯🚨 User Original Design Requirement (CRITICAL - MUST FOLLOW)
**Below is the user's core requirement. Your entire experimental plan MUST directly serve this goal:**

{initial_query}

⚠️ WARNING: Ensure that the chosen precursors, process parameters, and experimental conditions are all intended to fulfill the user requirement above!

---
"""
        
        # Fill Template
        prompt = user_goal_section + template.format(
            summary_report_json=json.dumps(summary_report_json, ensure_ascii=False),
            raw_features=json.dumps(raw_features, ensure_ascii=False, indent=2),
            raw_experiments=json.dumps(raw_experiments, ensure_ascii=False, indent=2),
            molecule_candidates_optimized=json.dumps(molecule_candidates_optimized[:10], ensure_ascii=False),
            process_params=json.dumps(process_params, ensure_ascii=False),
            recipe_strategy=process_params.get('Recipe_Strategy', 'Hydrothermal'),
            temperature=process_params.get('Temperature', '200 ℃'),
            time=process_params.get('Time', '8 h'),
            key_feature=key_feature
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2 
            )
            
            # Save Readable Log
            self._save_readable_log(messages, response)
            
            recipe_content = response.choices[0].message.content
        except Exception as e:
            print(f"❌ [Architect] LLM failed to generate report: {e}")
            return None
        
        # Save Report
        try:
            report_path = os.path.join(get_run_dir(), "Candidate_Recipe_Report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(recipe_content)
            print(f"✅ [Architect] Standard experimental plan generated: {report_path}")
        except Exception as e:
            print(f"❌ [Architect] Failed to save report file: {e}")

        return recipe_content