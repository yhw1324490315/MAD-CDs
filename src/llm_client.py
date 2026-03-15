import os
import json
import base64
import re
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load env
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)

class OpenAIResponseAdapter:
    """Adapts Gemini response to OpenAI-like response object"""
    def __init__(self, gemini_response):
        self.gemini_response = gemini_response
        
        # Prepare content
        self.content = gemini_response.text if hasattr(gemini_response, 'text') else ""
        
        # Prepare tool calls
        self.tool_calls = []
        if hasattr(gemini_response, 'function_calls'):
            for fc in gemini_response.function_calls:
                # Gemini FC has 'name' and 'args' (dict)
                # OpenAI expects 'function' object with 'name' and 'arguments' (string)
                self.tool_calls.append({
                    "id": "gemini_tool_call", # dummy id
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args)
                    },
                    "type": "function"
                })
        
        # Structure to mimic resp.choices[0].message
        class Message:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
                
            def to_dict(self):
                return {"content": self.content, "tool_calls": self.tool_calls}

        class Choice:
            def __init__(self, message):
                self.message = message

        self.choices = [Choice(Message(self.content, self.tool_calls if self.tool_calls else None))]
        
    def model_dump(self):
        # Rough emulation for logging
        return {"content": self.content, "tool_calls": self.tool_calls}


class GeminiClientWrapper:
    def __init__(self, api_key, model, temperature=0.1, thinking_level=None, media_resolution=None):
        # Use v1alpha for media_resolution support
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self.model = model
        self.temperature = temperature
        self.thinking_level = thinking_level
        self.media_resolution = media_resolution
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, model, messages, tools=None, tool_choice=None, response_format=None, temperature=None):
        """
        Mimics openai.chat.completions.create
        """
        
        # 1. Convert Messages
        system_instruction = None
        gemini_contents = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            parts = []
            if content:
                if isinstance(content, str):
                    parts.append(types.Part.from_text(text=content))
                elif isinstance(content, list):
                    # Handle multimodal content (text + image)
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item.get("text")))
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            # Parse data:image/png;base64,...
                            match = re.match(r"data:(.*?);base64,(.*)", url)
                            if match:
                                mime_type = match.group(1)
                                b64_data = match.group(2)
                                try:
                                    image_bytes = base64.b64decode(b64_data)
                                    blob = types.Blob(mime_type=mime_type, data=image_bytes)
                                    part_args = {"inline_data": blob}
                                    if self.media_resolution:
                                        part_args["media_resolution"] = {"level": self.media_resolution}
                                    parts.append(types.Part(**part_args))
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
            
            if role == "system":
                system_instruction = content
                
            elif role == "user":
                gemini_contents.append(types.Content(role="user", parts=parts))
                
            elif role == "assistant":
                # Handle tool calls
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        fname = func.get("name")
                        fargs = func.get("arguments")
                        if fname:
                            try:
                                args_dict = json.loads(fargs) if isinstance(fargs, str) else fargs
                            except:
                                args_dict = {}
                            parts.append(types.Part.from_function_call(name=fname, args=args_dict))
                
                if not parts:
                    # Provide empty text if nothing (Gemini might error on empty content)
                    parts.append(types.Part.from_text(text=" "))
                    
                gemini_contents.append(types.Content(role="model", parts=parts))
                
            elif role == "tool":
                # Fallback: Represent tool output as user text to ensure model sees it
                # Logic: OpenAI treats tool content as separate role. Gemini expects FunctionResponse.
                # Without easy access to function name (required for FunctionResponse), we use text fallback.
                text_content = f"Tool Execution Output (ID {msg.get('tool_call_id')}): {content}"
                gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text_content)])) 

        # 2. Tool Configuration
        gemini_tools = None
        if tools:
            # Convert OpenAI tools schema to Gemini tools
            # OpenAI: {"type": "function", "function": {...}}
            # Gemini: expects list of functions
            funcs = []
            for t in tools:
                if t.get("type") == "function":
                    f_def = t.get("function")
                    # Gemini (google-genai) might accept raw dicts or need helper construction.
                    # The v1 SDK allows passing dicts matching the schema.
                    funcs.append(f_def) # Hopefully compatible schema
            
            if funcs:
                gemini_tools = [types.Tool(function_declarations=funcs)]

        # 3. Response Format (JSON mode)
        mime_type = "text/plain"
        if response_format and response_format.get("type") == "json_object":
            mime_type = "application/json"

        # 4. Generate
        gen_config_args = {
            "temperature": temperature if temperature is not None else self.temperature,
            "system_instruction": system_instruction,
            "tools": gemini_tools,
            "response_mime_type": mime_type
        }
        
        if self.thinking_level:
             gen_config_args["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)

        config = types.GenerateContentConfig(**gen_config_args)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config
        )
        
        return OpenAIResponseAdapter(response)


class LLMClientFactory:
    _instance = None
    
    @staticmethod
    def get_client(config_loader):
        full_conf = config_loader.config
        llm_conf = full_conf.get('llm', {})
        provider = llm_conf.get('provider', 'openai')
        
        if provider == 'openai':
            conf = llm_conf.get('openai', {})
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=conf.get('base_url')
            ), conf.get('model_name'), conf.get('temperature', 0.1)
            
        elif provider == 'gemini':
            conf = llm_conf.get('gemini', {})
            api_key = os.getenv("GEMINI_API_KEY")
            model_name = conf.get('model_name', 'gemini-2.0-flash-exp')
            temp = conf.get('temperature', 0.1)
            thinking_level = conf.get('thinking_level')
            media_resolution = conf.get('media_resolution')
            
            return GeminiClientWrapper(api_key, model_name, temp, thinking_level, media_resolution), model_name, temp
            
        else:
            raise ValueError(f"Unknown provider: {provider}")

