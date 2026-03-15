import os
import yaml
import logging
from datetime import datetime

class ConfigLoader:
    _instance = None
    _config = None
    _prompts = None
    _run_dir = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.project_root, "config", "config.yaml")
        self.prompts_path = os.path.join(self.project_root, "config", "prompts.yaml")
        self._subdir = ""
        self.load_config()
        self.load_prompts()

    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Failed to load config.yaml: {e}")
            self._config = {}

    def load_prompts(self):
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Failed to load prompts.yaml: {e}")
            self._prompts = {}

    @property
    def config(self):
        return self._config

    @property
    def prompts(self):
        return self._prompts

    def get_llm_config(self):
        return self._config.get('llm', {})

    def get_data_path(self, key):
        path = self._config.get('data_paths', {}).get(key)
        if path and not os.path.isabs(path):
            path = os.path.join(self.project_root, path)
        return path
    
    def get_model_path(self, key):
        path = self._config.get('model_paths', {}).get(key)
        if path and not os.path.isabs(path):
            path = os.path.join(self.project_root, path)
        return path

    def set_run_dir(self, run_id=None):
        if run_id is None:
            run_id = datetime.now().strftime(self._config.get('output', {}).get('run_dir_format', "%Y-%m-%d_%H-%M-%S"))
        
        base_dir = self._config.get('output', {}).get('base_dir', 'experiments')
        # Handle relative path for base_dir
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(self.project_root, base_dir)
            
        self._run_dir = os.path.join(base_dir, run_id)
        os.makedirs(self._run_dir, exist_ok=True)
        print(f"📂 Run directory initialized: {self._run_dir}")
        return self._run_dir
    
    def set_subdir(self, subdir):
        self._subdir = subdir

    @property
    def run_dir(self):
        if self._run_dir is None:
            # If not explicitly set, create a default one
            self.set_run_dir()
            
        if self._subdir:
            d = os.path.join(self._run_dir, self._subdir)
            os.makedirs(d, exist_ok=True)
            return d
            
        return self._run_dir

    @property
    def base_run_dir(self):
        if self._run_dir is None:
            self.set_run_dir()
        return self._run_dir

# Global helper to get valid run directory
def get_run_dir():
    return ConfigLoader.get_instance().run_dir

def get_base_run_dir():
    return ConfigLoader.get_instance().base_run_dir

def set_run_subdir(subdir):
    ConfigLoader.get_instance().set_subdir(subdir)

import threading
_log_lock = threading.Lock()

def log_to_global_file(agent_name, input_content, output_content, step_info=""):
    """
    Thread-safe logging to a single global file.
    Handles both string inputs and message objects/dicts.
    """
    try:
        base_dir = get_base_run_dir()
        log_path = os.path.join(base_dir, "Full_Interaction_Chain_Log.txt")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure input_content is string
        if not isinstance(input_content, str):
            input_content = str(input_content)
        
        # Ensure output_content is string
        if not isinstance(output_content, str):
            output_content = str(output_content)
        
        # Truncate if too long
        if len(input_content) > 50000:
            input_content = input_content[:50000] + "\n... [TRUNCATED]"
        if len(output_content) > 50000:
            output_content = output_content[:50000] + "\n... [TRUNCATED]"
        
        entry = (
            f"\n{'='*60}\n"
            f"TIMESTAMP: {timestamp}\n"
            f"AGENT: {agent_name}\n"
            f"CONTEXT: {step_info}\n"
            f"{'-'*20} INPUT {'-'*20}\n"
            f"{input_content}\n"
            f"{'-'*20} OUTPUT {'-'*20}\n"
            f"{output_content}\n"
            f"{'='*60}\n"
        )
        
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)
                
    except Exception as e:
        print(f"⚠️ Global logging failed: {e}")

# Global helper to load prompt
def get_prompt(key, default=""):
    return ConfigLoader.get_instance().prompts.get(key, default)

# Global helper to get LLM client config
def get_llm_config():
    return ConfigLoader.get_instance().get_llm_config()

def get_llm_client():
    from src.llm_client import LLMClientFactory
    return LLMClientFactory.get_client(ConfigLoader.get_instance())
