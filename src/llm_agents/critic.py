import os
import json
import time
import re
import concurrent.futures
from openai import OpenAI
from src.utils import ConfigLoader, log_to_global_file
from src.llm_client import GeminiClientWrapper

class CriticAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        
        # Load judges from config.yaml
        critic_config = self.config_loader.config.get('critic', {})
        self.judges = critic_config.get('judges', [])
        
        if not self.judges:
            print("⚠️ [CriticAgent] No judges configured in config.yaml, using defaults.")
            self.judges = [
                {"name": "Default", "provider": "openai", "model": "gpt-4o", 
                 "api_key_env": "OPENAI_API_KEY", "base_url": "", "temperature": 0.1}
            ]
        
        print(f"📋 [CriticAgent] Loaded {len(self.judges)} judge model configurations")
        for j in self.judges:
            print(f"   - {j.get('name')}: {j.get('model')} @ {j.get('base_url', 'default')}")
            
        self.shap_knowledge = self.config_loader.prompts.get('shap_knowledge', '')

    def evaluate(self, architect_recipe, summary_report, iteration=1, log_dir="experiments", initial_query=None):
        # Create output directory for this iteration
        review_dir = os.path.join(log_dir, "Critic_Reviews", f"Round_{iteration}")
        os.makedirs(review_dir, exist_ok=True)
        
        print(f"\n⚖️ [CriticAgent] Convening a panel of {len(self.judges)} LLM judges to review the recipe (Round {iteration})...")
        if initial_query:
            print(f"🎯 User original requirement: {initial_query}")
        print(f"📂 Review details will be saved to: {review_dir}")
        
        results = []
        
        user_goal_reminder = ""
        if initial_query:
            user_goal_reminder = f"""
### 🎯🚨 User Original Design Requirement (CRITICAL - MUST VERIFY)
**Below is the user's core requirement. When judging if the plan is reasonable, you MUST first verify if the plan serves this goal:**

{initial_query}

⚠️ If the experimental plan deviates from the user's original requirement (e.g., user requested "design a material with long lifetime", but the plan optimizes "emission wavelength"), you must strictly deduct points and point out the issue!

---
"""
        
        def call_judge(judge_config):
            # Read individual judge configuration
            judge_name = judge_config.get("name", "Unknown")
            model_name = judge_config.get("model", "Gemini-3-Pro")
            provider = judge_config.get("provider", "openai")
            api_key_env = judge_config.get("api_key_env", "OPENAI_API_KEY")
            base_url = judge_config.get("base_url", "")
            temperature = judge_config.get("temperature", 0.1)
            
            # Get API key from environment
            api_key = os.getenv(api_key_env)
            if not api_key:
                print(f"⚠️ [{judge_name}] API key not found in env: {api_key_env}")
                return {
                    "judge_model": judge_name,
                    "is_reasonable": False,
                    "score": 0,
                    "reason": f"API key not configured: {api_key_env}"
                }
            
            prompt = f"""
            You are a strict materials science review expert. Your task is to first think about the relevant knowledge of carbon dot-based long persistent luminescence materials according to the task requirements raised by the user, then combine knowledge and the following information to judge whether the [Experimental Plan] is reasonable or not, and give a score (0-10).

{user_goal_reminder}
### SHAP Rules
{self.shap_knowledge}

### Task Summary
{json.dumps(summary_report, ensure_ascii=False)}

### Plan to be Reviewed
{architect_recipe}

---
            ### Evaluation Criteria
            1. 🚨 **MOST IMPORTANT**: First verify whether the experimental plan directly serves the user's original design requirement. If the plan's goal does not match the user's requirement (e.g., user asks for long lifetime, plan optimizes emission wavelength), directly deem it unreasonable (0-3 points).
            2. Strictly compare if the [Precursor Selection], [Temperature], [Solvent] in the experimental plan comply with the SHAP rules and the suggestions in the data summary.
            3. If the plan chose a "negatively correlated" parameter to "increase" the target value, it must be deemed unreasonable (0-5 points).
            4. If the plan ignored key Bit features or gate rules (e.g., MW limit), it must be deemed unreasonable (0-5 points).
            5. If the plan has rigorous logic and conforms to all rules, give a high score (8-10 points).
            6. If there are issues, reasonable correction suggestions must be provided.
            7. You must strictly judge the reasonableness of precursors and experimental conditions; scoring must be reasonable!
            
            ### Output strictly as JSON:
{{
    "judge_model": "{judge_name}",
    "is_reasonable": true/false,
    "score": 5.5, // 0-10 points, allows decimals
    "reason": "Detailed reason pointing out exactly where it matched or violated rules",
    "suggest": "Detailed suggestions based on user requirements"
}}
```"""
            
            content = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    if provider == 'openai':
                        # Use individual base_url for this judge
                        client = OpenAI(
                            api_key=api_key, 
                            base_url=base_url if base_url else None
                        )
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=4096  # Ensure complete response for thinking models
                        )
                        content = response.choices[0].message.content
                        
                    elif provider == 'gemini':
                        gemini_conf = self.config_loader.config.get('llm', {}).get('gemini', {})
                        client = GeminiClientWrapper(
                            api_key=api_key, 
                            model=model_name, 
                            temperature=temperature,
                            thinking_level=gemini_conf.get('thinking_level'),
                            media_resolution=gemini_conf.get('media_resolution')
                        )
                        response = client.create(
                             model=model_name,
                             messages=[{"role": "user", "content": prompt}]
                        )
                        content = response.choices[0].message.content
                    
                    if content:
                        break # Success
                        
                except Exception as e:
                    print(f"⚠️ Judge {judge_name} Connection Error (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        return {
                            "judge_model": judge_name,
                            "is_reasonable": False,
                            "score": 0,
                            "reason": f"Connection Error after {max_retries} retries: {str(e)}"
                        }

            # --- Save Readable Interaction Log for verify (Consolidated) ---
            if content:
                log_to_global_file(
                      f"CriticAgent::{judge_name}", 
                      prompt, 
                      content, 
                      f"Critic Review Iteration {iteration}"
                )
            
            # ====================================================================================
            #                          ROBUST JSON PARSING
            # ====================================================================================
            if not content:
                return {
                    "judge_model": judge_name,
                    "is_reasonable": False,
                    "score": 0,
                    "reason": "Empty response from LLM"
                }

            # 1. Try to parse directly (Happy Path)
            try:
                return json.loads(content)
            except:
                pass

            # 2. Markdown extraction strategy for thinking models
            # Strategy: Find all code blocks, traverse backwards 
            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(code_block_pattern, content)
            
            if matches:
                for match_content in reversed(matches):
                    try:
                        parsed = json.loads(match_content.strip())
                        # Simple structural validation
                        if isinstance(parsed, dict) and ('score' in parsed or 'is_reasonable' in parsed):
                            return parsed
                    except:
                        continue

            # 3. Ultimate Fallback: Balanced Braces lookup
            try:
                open_braces = [m.start() for m in re.finditer(r'\{', content)]
                # Search backwards, prioritize JSON at the end
                for start in reversed(open_braces):
                    depth = 0
                    for i in range(start, len(content)):
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                            if depth == 0:
                                candidate = content[start:i+1]
                                try:
                                    parsed = json.loads(candidate)
                                    if isinstance(parsed, dict) and ('score' in parsed or 'is_reasonable' in parsed):
                                        return parsed
                                except:
                                    pass
                                break 
            except Exception as e:
                # Tracing disabled to prevent terminal clutter
                pass

            # If all parsing fails, return error dict
            tail_content = content[-200:].replace('\n', ' ')
            print(f"⚠️ Judge {judge_name} failed all JSON parsing methods. Tail: {tail_content}")
            
            return {
                "judge_model": judge_name,
                "is_reasonable": False,
                "score": 0,
                "reason": f"Could not parse JSON from response. Response tail: {tail_content}"
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            futures = [executor.submit(call_judge, judge) for judge in self.judges]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results.append(res)
                # Save individual review JSON immediately
                safe_name = res.get('judge_model', 'unknown').replace('/', '_').replace(':', '')
                with open(os.path.join(review_dir, f"Review_{safe_name}.json"), "w", encoding='utf-8') as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
        
        # Statistics
        pass_count = sum(1 for r in results if r.get('is_reasonable') == True)
        scores = [r.get('score', 0) for r in results if isinstance(r.get('score'), (int, float))]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"\n📊 Review Results: {pass_count}/7 Passed | Average Score: {avg_score:.1f}")
        
        for r in results:
            icon = "✅" if r.get('is_reasonable') else "❌"
            print(f"   {icon} [{r.get('judge_model')}]: {r.get('score')} pts - {r.get('reason')}")

        # Save Round Summary
        summary_stats = {
            "iteration": iteration,
            "pass_count": pass_count,
            "fail_count": 7 - pass_count,
            "avg_score": avg_score,
            "details": results
        }
        with open(os.path.join(review_dir, "Round_Summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)

        return {
            "pass": avg_score >= 8.5, 
            "pass_count": pass_count,
            "avg_score": avg_score,
            "details": results
        }