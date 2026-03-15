# -*- coding: utf-8 -*-
import json
import os
import re
from dotenv import load_dotenv
from src.utils import get_run_dir, set_run_subdir, get_base_run_dir, ConfigLoader

from src.llm_agents.planner import PlannerAgent
from src.llm_agents.deep_analysis_tool import DeepAnalysisRunner
from src.llm_agents.summary import SummaryAgent
from src.llm_agents.architect import ArchitectAgent

# Get project root (this file is in root)
project_root = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)

# =========================================================
# [User Configuration]
# =========================================================
# Retrieval data size (123241653 is full dataset)
# Recommend testing with smaller size first, e.g., 5000, then change to 123241653 or None
MOLECULE_SEARCH_LIMIT = 123241653
# Maximum molecular weight limit (molecules exceeding this are omitted)
MOLECULE_MAX_MW = 500           
# =========================================================

def clean_json_str(content):
    """
    Clean the JSON string output from LLM, removing Markdown code block tags.
    """
    if not isinstance(content, str):
        return "{}"
        
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content

from src.llm_agents.critic import CriticAgent

def main():
    # ==========================================
    # Print total CID-SMILES count
    # ==========================================
    try:
        conf = ConfigLoader.get_instance()
        cid_path = conf.get_data_path('cid_smiles')
        print(f"📊 Counting total molecules in CID-SMILES library (Path: {cid_path})...")
        if cid_path and os.path.exists(cid_path):
            with open(cid_path, 'rb') as f:
                total_molecules = 0
                while True:
                    buffer = f.read(1024 * 1024 * 4) # 4MB chunks
                    if not buffer:
                        break
                    total_molecules += buffer.count(b'\n')
            print(f"🔢 Total CID-SMILES molecules: {total_molecules:,}")
        else:
            print(f"⚠️ Cannot find data file: {cid_path}")
    except Exception as e:
        print(f"⚠️ Error counting total molecules: {e}")

    print("🚀 === [CD-LPL Autonomous Discovery System] Full Pipeline Integration Test (Architecture with Critic Loop) ===\n")
    print(f"⚙️  Configuration: Search Limit = {MOLECULE_SEARCH_LIMIT:,} | Max MW = {MOLECULE_MAX_MW} Da\n")

    initial_query = "Help me design a carbon dot-based long persistent luminescence material with an afterglow emission wavelength over 750nm."

    current_query = initial_query
    
    max_iterations = 5
    iteration = 0
    score_history = []
    # Initialize Experiment Root Directory
    base_experiment_dir = get_base_run_dir()
    print(f"📂 Experiment Root Dir: {base_experiment_dir}")
    
    last_critic_feedback = None
    rejected_history = [] 
    
    while iteration < max_iterations:
        iteration += 1
        
        # Create Iteration Subdirectory
        set_run_subdir(f"Iteration_{iteration}")
        print(f"\n🔄 [Iteration {iteration}/{max_iterations}] Starting new design cycle...")
        print(f"📂 Output directory for this iteration: {get_run_dir()}")
        
        # =========================================================================
        # Step 1: Planner Agent
        # =========================================================================
        print("---------------------------------------------------------------")
        print("🧠 [Step 1] Planner Agent")
        print("---------------------------------------------------------------")
        planner = PlannerAgent()
        print_query = current_query[:100] + "..." if len(current_query) > 100 else current_query
        print(f"📥 Input prompt: {print_query}")
        print(f"🎯 User original requirement: {initial_query}")
        
        # Prepend user original requirement to ensure Planner does not drift
        query_with_goal = f"[🎯 User Original Design Requirement (MUST FOLLOW)]\n{initial_query}\n\n---\n\n{current_query}"
        
        try:
            planner_json_str, planner_context = planner.run(query_with_goal)
        except Exception as e:
            print(f"❌ Planner execution error: {e}")
            break

        try:
            planner_data = json.loads(clean_json_str(planner_json_str))
            print(f"\n📋 Planner Planning Result: ")
            print(json.dumps(planner_data, indent=4, ensure_ascii=False))
        except Exception as e:
            print(f"❌ Planner JSON parsing failed: {e}")
            print(f"Raw output: {planner_json_str}")
            break

        # =========================================================================
        # Step 2: Deep Analysis
        # =========================================================================
        if "key_bits_to_decode" in planner_data and planner_data["key_bits_to_decode"]:
            print("\n---------------------------------------------------------------")
            print("⛏️ [Step 2] Deep Analysis")
            print("---------------------------------------------------------------")
            
            analyzer = DeepAnalysisRunner()
            analysis_result = analyzer.analyze(planner_data)
            status = analysis_result.get('status')
            output_dirs = analysis_result.get('output_dirs', [])
            
            if (status == 'success' or status == 'partial_success') and output_dirs:
                # =========================================================================
                # Step 3: Summary Agent
                # =========================================================================
                print("\n---------------------------------------------------------------")
                print("👁️ [Step 3] Summary Agent")
                print("---------------------------------------------------------------")
                
                try:
                    summarizer = SummaryAgent()
                    # Pass original query to SummaryAgent
                    summary_report = summarizer.summarize(planner_data, output_dirs, critic_feedback=last_critic_feedback, initial_query=initial_query)
                    
                    if not summary_report or "error" in summary_report:
                        print("❌ Summary generation failed.")
                        break
                    
                    # =========================================================================
                    # Step 4: Architect Agent
                    # =========================================================================
                    print("\n---------------------------------------------------------------")
                    print("🏗️ [Step 4] Architect Agent")
                    print("---------------------------------------------------------------")
                    
                    architect = ArchitectAgent()
                    # Pass original query to ArchitectAgent
                    recipe_content = architect.generate_recipe(
                        summary_report_json=summary_report, 
                        planner_context=planner_context,
                        scout_limit=MOLECULE_SEARCH_LIMIT,
                        scout_max_mw=MOLECULE_MAX_MW,
                        initial_query=initial_query
                    )
                    
                    if not recipe_content:
                        print("❌ Architect recipe generation failed.")
                        break
                        
                    print("✅ Candidate Recipe Generated.")

                    # =========================================================================
                    # Step 5: Critic Agent (Review)
                    # =========================================================================
                    print("\n---------------------------------------------------------------")
                    print("⚖️ [Step 5] Critic Agent")
                    print("---------------------------------------------------------------")
                    
                    critic = CriticAgent()
                    # Pass iteration, log_dir and initial_query for versioned logging
                    review_result = critic.evaluate(recipe_content, summary_report, iteration=iteration, log_dir=get_run_dir(), initial_query=initial_query)
                    
                    # Track Score
                    avg_score = review_result.get('avg_score', 0)
                    score_history.append({
                        "iteration": iteration,
                        "avg_score": avg_score,
                        "pass_count": review_result.get('pass_count', 0),
                        "passed": review_result['pass']
                    })
                    
                    # Save Score History to Base Directory
                    history_path = os.path.join(get_base_run_dir(), "Score_History.json")
                    try:
                         with open(history_path, "w", encoding="utf-8") as f:
                            json.dump(score_history, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"⚠️ Failed to save score history: {e}")

                    current_trend = [f"{s['avg_score']:.1f}" for s in score_history]
                    print(f"📈 Current score trend: {current_trend}")
                    
                    if review_result['pass']:
                        print("\n" + "="*60)
                        print("🎉  Recipe passed review! (Pass count: {}/7)".format(review_result['pass_count']))
                        print("="*60)
                        
                        # Save Final Report
                        with open("Final_Approved_Recipe.md", "w", encoding="utf-8") as f:
                            f.write(recipe_content)
                        break # Exit loop (Success)
                    
                    else:
                        print(f"\n⚠️ Recipe failed review (Pass count: {review_result['pass_count']}/7). Entering next iteration.")
                        
                        # 1. Structure feedback for this round
                        current_round_feedback = f"Feedback for Iteration {iteration}:\n"
                        for res in review_result['details']:
                            current_round_feedback += f"- [{res['judge_model']}]: {res.get('score')} points - {'Valid' if res.get('is_reasonable') else 'Invalid'} - {res.get('reason')}\n"
                        
                        # 2. Add to history
                        rejected_history.append({
                            "iteration": iteration,
                            "recipe": recipe_content,
                            "feedback": current_round_feedback,
                            "summary_context": json.dumps(summary_report, ensure_ascii=False)
                        })

                        # 3. Consolidate total feedback
                        consolidated_feedback = "[⚠️ Cumulative Failure History]\n"
                        consolidated_feedback += "The following experimental plans were REJECTED in previous iterations. You MUST verify why they failed and avoid repeating similar mistakes.\n\n"
                        
                        for hist in rejected_history:
                            consolidated_feedback += f"=== Rejected Round {hist['iteration']} ===\n"
                            consolidated_feedback += f"[Recipe Preview]: {hist['recipe'][:300]}...\n"
                            consolidated_feedback += f"[Reviewer Feedback]:\n{hist['feedback']}\n"
                            consolidated_feedback += "-"*40 + "\n"

                        current_query = consolidated_feedback # Update query for Planner in next loop
                        last_critic_feedback = consolidated_feedback # Store for SummaryAgent in next loop

                except Exception as e:
                    print(f"❌ Pipeline execution error: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                 print("❌ Deep Analysis Failed.")
                 break
        else:
             print("⚠️ No key bits to decode.")
             break
    
    if iteration >= max_iterations:
        print("\n❌ Reached maximum iterations, failed to generate a recipe passing review.")

    print(f"\n🏁 Pipeline ended. Final score history: {[s['avg_score'] for s in score_history]}")

if __name__ == "__main__":
    main()
