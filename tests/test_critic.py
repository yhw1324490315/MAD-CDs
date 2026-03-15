"""
CriticAgent Test Script
Used to verify if each configured judge works properly
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)

from src.llm_agents.critic import CriticAgent

def test_critic_agent():
    print("=" * 60)
    print("🧪 CriticAgent Test Script")
    print("=" * 60)
    
    # Initialize CriticAgent
    print("\n📋 Initializing CriticAgent...")
    try:
        critic = CriticAgent()
        print("✅ CriticAgent initialization successful")
    except Exception as e:
        print(f"❌ CriticAgent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show configured judge models
    print(f"\n📊 Configured {len(critic.judges)} judge models:")
    for i, judge in enumerate(critic.judges, 1):
        print(f"   {i}. {judge.get('name', 'Unknown')}")
        print(f"      Model: {judge.get('model')}")
        print(f"      API: {judge.get('base_url', 'default')}")
        print(f"      Key: {judge.get('api_key_env')}")
        
        # Check if API key exists
        key_env = judge.get('api_key_env', 'OPENAI_API_KEY')
        key_value = os.getenv(key_env)
        if key_value:
            print(f"      ✅ API Key configured ({key_env})")
        else:
            print(f"      ❌ API Key not found ({key_env})")
        print()
    
    # Prepare test data
    test_recipe = """
    ## Experimental Plan Test
    
    ### Precursor Selection
    - Precursor 1: Citric Acid, MW = 192 Da
    - Precursor 2: Ethylenediamine (EDA), MW = 60 Da
    
    ### Synthesis Conditions
    - Temperature: 180°C
    - Time: 6 hours
    - Solvent: Deionized water
    
    ### Expected Performance
    - Emission Wavelength: 600-700nm
    """
    
    test_summary = {
        "target_property": "emission",
        "task_analysis": "Design a carbon dot material emitting near-infrared light",
        "critical_features_analysis": [
            {
                "feature_name": "Pre1_Molecular Weight",
                "type": "Condition",
                "impact_trend": "Positive correlation",
                "design_recommendation": "Select precursor with MW > 200Da"
            }
        ]
    }
    
    # Ask whether to run full test
    print("\n" + "=" * 60)
    print("🔧 Test Options:")
    print("   1. Test a single model (Fast)")
    print("   2. Test all models (Full)")
    print("   3. Exit")
    print("=" * 60)
    
    choice = input("\nPlease select (1/2/3): ").strip()
    
    if choice == "1":
        # Single model test
        print("\nAvailable models:")
        for i, judge in enumerate(critic.judges, 1):
            print(f"   {i}. {judge.get('name')}")
        
        idx = input(f"\nPlease enter model number (1-{len(critic.judges)}): ").strip()
        try:
            idx = int(idx) - 1
            if 0 <= idx < len(critic.judges):
                selected_judge = critic.judges[idx]
                print(f"\n🧪 Currently testing: {selected_judge.get('name')}...")
                
                # Temporarily keep only the selected model
                original_judges = critic.judges
                critic.judges = [selected_judge]
                
                try:
                    result = critic.evaluate(test_recipe, test_summary, iteration=0, log_dir="test_output")
                    print("\n📊 Test Results:")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"\n❌ Test Failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    critic.judges = original_judges
            else:
                print("❌ Invalid number")
        except ValueError:
            print("❌ Please enter a number")
            
    elif choice == "2":
        # Full test
        print("\n🧪 Testing all models...")
        print("⏳ This may take a few minutes...\n")
        
        try:
            result = critic.evaluate(test_recipe, test_summary, iteration=0, log_dir="test_output")
            
            print("\n" + "=" * 60)
            print("📊 Full Test Results:")
            print("=" * 60)
            print(f"Passed count: {result.get('pass_count', 0)}/{len(critic.judges)}")
            print(f"Average score: {result.get('avg_score', 0):.1f}")
            print(f"Passed overall: {'✅ Yes' if result.get('pass') else '❌ No'}")
            
            print("\n📋 Detailed results per model:")
            for detail in result.get('details', []):
                icon = "✅" if detail.get('is_reasonable') else "❌"
                print(f"\n{icon} [{detail.get('judge_model', 'Unknown')}]")
                print(f"   Score: {detail.get('score', 0)}")
                print(f"   Reason: {detail.get('reason', 'N/A')[:100]}...")
                
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("👋 Exiting test")
        return
    
    print("\n" + "=" * 60)
    print("🏁 Test Completed")
    print("=" * 60)

if __name__ == "__main__":
    test_critic_agent()
