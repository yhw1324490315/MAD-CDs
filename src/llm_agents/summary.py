import os
import json
import base64
import glob
import pandas as pd
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client, log_to_global_file

from PIL import Image
import io

# SVG to PNG conversion - try multiple backends
HAS_SVG_CONVERTER = False
SVG_BACKEND = None

# Try 1: svglib + reportlab (needs Cairo on some systems)
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    HAS_SVG_CONVERTER = True
    SVG_BACKEND = "svglib"
except Exception as e:
    pass

# Try 2: wand (ImageMagick binding)
if not HAS_SVG_CONVERTER:
    try:
        from wand.image import Image as WandImage
        HAS_SVG_CONVERTER = True
        SVG_BACKEND = "wand"
    except Exception:
        pass

# Fallback message handled inside methods to avoid multiprocessing spam

# Load environment variables from the project config directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)

class SummaryAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        
        # Initialize LLM Client
        self.client, self.model, self.temperature = get_llm_client()

        # Load System Prompt from yaml
        self.system_prompt = get_prompt('summary_agent_system', '')

    # ================= Tool Functions =================

    def _encode_image(self, image_path):
        """Convert local image to Base64"""
        if not os.path.exists(image_path): return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except: return None

    def _read_file_content(self, file_path):
        if not os.path.exists(file_path): return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except: return None

    def _get_browser_executable(self):
        """Find Edge or Chrome browser path in system"""
        paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def _convert_svg_using_browser(self, svg_path, png_path):
        """Use system browser screenshot to convert SVG to PNG (Ultimate Fallback)"""
        import subprocess
        import shutil
        import time
        
        browser_exe = self._get_browser_executable()
        if not browser_exe:
            print("⚠️ Edge or Chrome browser not found, cannot perform fallback conversion.")
            return False
            
        try:
            # Ensure absolute paths
            abs_svg_path = os.path.abspath(svg_path)
            abs_png_path = os.path.abspath(png_path)
            
            # Browser generated default screenshot filename is usually "screenshot.png" in current directory
            # Or we can specify --screenshot=path (supported in newer versions)
            
            # Set window size to match SVG (usually 400x400)
            cmd = [
                browser_exe,
                "--headless",
                "--disable-gpu",
                "--window-size=500,500",
                "--hide-scrollbars",
                f"--screenshot={abs_png_path}",
                f"file:///{abs_svg_path.replace(os.sep, '/')}"
            ]
            
            # Run browser command
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            # Check if file was generated
            if os.path.exists(abs_png_path) and os.path.getsize(abs_png_path) > 0:
                return True
                
            # Try to find default screenshot.png (if --screenshot arg doesn't support specific path)
            cwd_screenshot = os.path.join(os.getcwd(), "screenshot.png")
            if os.path.exists(cwd_screenshot):
                shutil.move(cwd_screenshot, abs_png_path)
                return True
                
            print(f"⚠️ Browser screenshot failed, output file not found. Stderr: {result.stderr.decode('utf-8', errors='ignore')}")
            return False
            
        except Exception as e:
            print(f"⚠️ Browser conversion exception: {e}")
            return False

    def _svg_to_png(self, svg_path, png_path=None):
        """
        Convert SVG file to PNG and return Base64 string.
        Strategy priority:
        1. svglib/wand (if library is available)
        2. System browser screenshot (Universal Fallback)
        """
        if not os.path.exists(svg_path):
            print(f"⚠️ SVG file not found: {svg_path}")
            return None
        
        # If PNG path not specified, save in same directory as SVG
        if png_path is None:
            png_path = svg_path.replace('.svg', '.png')
        
        conversion_success = False

        # Strategy 1: Python libraries
        if HAS_SVG_CONVERTER:
            try:
                if SVG_BACKEND == "svglib":
                    drawing = svg2rlg(svg_path)
                    if drawing:
                        renderPM.drawToFile(drawing, png_path, fmt="PNG")
                        conversion_success = True
                elif SVG_BACKEND == "wand":
                    with WandImage(filename=svg_path) as img:
                        img.format = 'png'
                        img.save(filename=png_path)
                        conversion_success = True
            except Exception as e:
                print(f"⚠️ Python library conversion failed ({SVG_BACKEND}): {e}")
        
        # Strategy 2: Browser screenshot (if strategy 1 fails or unavailable)
        if not conversion_success:
            print(f"🔄 Attempting to convert using system browser: {os.path.basename(svg_path)}")
            conversion_success = self._convert_svg_using_browser(svg_path, png_path)
            
        if conversion_success and os.path.exists(png_path):
            try:
                # Read and encode to Base64
                with open(png_path, 'rb') as f:
                    png_data = f.read()
                base64_str = base64.b64encode(png_data).decode('utf-8')
                print(f"✅ SVG successfully converted to PNG: {png_path}")
                return base64_str
            except Exception as e:
                 print(f"⚠️ Failed to read PNG: {e}")
                 return None
        else:
            print(f"❌ Could not convert SVG to PNG (due to missing environment dependencies)")
            return None


    def _save_readable_log(self, input_messages, raw_response, step_info="Summary"):
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
             
             log_to_global_file("SummaryAgent", input_str, resp_content, step_info)
        except Exception as e:
             print(f"⚠️ Log save failed: {e}")

    # ================= Core Analysis Module =================

    def summarize(self, planner_json, output_dirs, critic_feedback=None, initial_query=None):
        print("🤖 [SummaryAgent] Starting summary analysis...")
        if initial_query:
            print(f"🎯 User original requirement: {initial_query}")
        
        evidence_text = ""
        evidence_count = 0
        evidence_list = []  # Store detailed info for each piece of evidence
        svg_images = []  # Store SVG image info for LLM visual input

        # Collect Evidence from analysis directories (RECURSIVE SEARCH)
        for d in output_dirs:
            if not os.path.exists(d): continue
            
            # 1. First collect SVG image directories (molecular structures), generate PNG from SMILES in source_info.txt
            svg_pattern = os.path.join(d, "**", "*.svg")
            svg_files = glob.glob(svg_pattern, recursive=True)
            for svg_file in svg_files:
                svg_dir = os.path.dirname(svg_file)
                svg_info = {
                    "file_path": svg_file,
                    "file_name": os.path.basename(svg_file),
                    "rel_path": os.path.relpath(svg_file, d),
                    "source_dir": os.path.basename(d)
                }
                
                # Attempt to read corresponding source_info.txt to get SMILES info
                source_info_path = os.path.join(svg_dir, "source_info.txt")
                if os.path.exists(source_info_path):
                    try:
                        with open(source_info_path, 'r', encoding='utf-8') as f:
                            svg_info["source_info"] = f.read()
                    except Exception as e:
                        print(f"⚠️ Failed to read source_info: {e}")
                
                # Directly convert SVG to PNG (SVG already generated by deep_analysis_tool.py)
                png_base64 = self._svg_to_png(svg_file)
                if png_base64:
                    svg_info["png_base64"] = png_base64
                
                # Even if image conversion fails, add info (including SMILES text) for fallback handling
                svg_images.append(svg_info)
            
            # 2. Recursively collect all text/csv/md files
            for ext in ["*.txt", "*.csv", "*.md"]:
                pattern = os.path.join(d, "**", ext)
                files = glob.glob(pattern, recursive=True)
                for f in files:
                    try:
                        with open(f, "r", encoding="utf-8") as file:
                            content = file.read()
                            # Truncate overly long content to prevent context overflow
                            if len(content) > 5000: 
                                content = content[:5000] + "\n...(truncated)..."
                            
                            # Get relative path for better context
                            rel_path = os.path.relpath(f, d)
                            source_label = f"{os.path.basename(d)}/{rel_path}"
                            
                            evidence_text += f"\n\n### Evidence Source: {source_label}\n"
                            evidence_text += content
                            evidence_count += 1
                            
                            # Add to evidence list
                            evidence_list.append({
                                "index": evidence_count,
                                "source": source_label,
                                "file_path": f,
                                "content_length": len(content)
                            })
                    except Exception as e:
                        print(f"Error reading {f}: {e}")

        # ========== Print detailed evidence material list ==========
        print("\n" + "="*70)
        print(f"📊 Summary Agent collected {evidence_count} evidence files + {len(svg_images)} molecular structure images")
        print("="*70)
        
        print("\n📄 [Text Evidence Material Details]:")
        print("-"*70)
        for ev in evidence_list:
            print(f"  [{ev['index']:02d}] {ev['source']}")
            print(f"       Path: {ev['file_path']}")
            print(f"       Content Length: {ev['content_length']} chars")
        print("-"*70)
        
        print(f"\n🖼️  [Molecular Structure Images (SVG) Details]:")
        print("-"*70)
        for i, img in enumerate(svg_images, 1):
            print(f"  [{i:02d}] {img['file_name']}")
            print(f"       Path: {img['file_path']}")
            if 'source_info' in img:
                # Extract SMILES info
                lines = img['source_info'].split('\n')
                for line in lines:
                    if 'SMILES' in line or 'Bit' in line:
                        print(f"       {line.strip()}")
            has_png = '✅ PNG Converted' if 'png_base64' in img else '❌ Conversion Failed'
            print(f"       Status: {has_png}")
        print("-"*70)
        print("="*70 + "\n")

        # Construct Prompt - Ensure user original requirement is added first
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # 🚨 KEY: Emphasize user's original design requirement at the very beginning
        if initial_query:
            messages.append({
                "role": "user",
                "content": f"[🎯🚨 User Original Design Requirement (CRITICAL - MUST NOT DEVIATE)]\nPlease always keep in mind and strictly base all analysis and recommendations on the following user requirement:\n\n{initial_query}\n\n⚠️ WARNING: All your analysis and design recommendations must directly serve the above user requirement and must not deviate!"
            })
        
        messages.append({"role": "user", "content": f"Here is the Planner Agent's original planning task:\n{json.dumps(planner_json, ensure_ascii=False)}"})

        # Inject Critic Feedback if available (Iterative Refinement)
        if critic_feedback:
             messages.append({
                "role": "user", 
                "content": f"[⚠️ Reasons for Failure in Previous Review (Critical Feedback)]\nPlease highly prioritize the following review feedback and make targeted corrections in this design:\n{critic_feedback}"
            })

        # Inject Global SHAP Knowledge
        shap_knowledge = self.config_loader.prompts.get('shap_knowledge')
        if shap_knowledge:
             messages.append({
                "role": "user", 
                "content": f"[Global Feature Importance Knowledge Base (Global SHAP Knowledge)]\nPlease make sure to refer to the following global rules, this is the highest priority rule based on full-dataset data mining:\n{shap_knowledge}"
            })
        
        # Inject Evidence (Text)
        if evidence_text:
            messages.append({
                "role": "user", 
                "content": f"Here is the specific evidence mined by Deep Analysis:\n{evidence_text}"
            })
        else:
             messages.append({
                "role": "user", 
                "content": "No new quantitative analysis evidence was mined this time, please reason primarily based on the knowledge base and planning intent."
            })
        
        # 🖼️ Inject Structure Evidence (Images OR Text)
        if svg_images:
            print(f"\n🖼️ Sending {len(svg_images)} critical structural evidences to LLM...")
            
            for img_info in svg_images:
                # Construct base description
                img_desc = f"[Critical Substructure Evidence]\nFile: {img_info['file_name']}"
                if 'source_info' in img_info:
                    img_desc += f"\n{img_info['source_info']}"
                
                # Method A: Has PNG image, use Vision capability
                if 'png_base64' in img_info:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"{img_desc}\n\n"
                                    f"⚠️ **IMPORTANT NOTICE**:\n"
                                    f"1. **'Substructure SMILES'** represents the core substructure of this Bit fingerprint, prioritize this.\n"
                                    f"2. **'Source Molecule SMILES'** is merely an example parent molecule containing this Bit, provided for context.\n"
                                    f"3. The **highlighted (red/colored)** portion in the image corresponds to the Substructure, while the grey background is the parent molecule.\n"
                                    f"4. Focus heavily on analyzing the chemical characteristics of the **Substructure (highlighted)** and its contribution to performance. Do not mistake the entire parent molecule as the definition of this feature.\n"
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_info['png_base64']}"
                                }
                            }
                        ]
                    })
                # Method B: **Fallback** no image, send text prompt only
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"{img_desc}\n\n"
                            f"⚠️ (Note: Cannot generate preview image due to environmental constraints)\n"
                            f"Please distinguish carefully:\n"
                            f"- **Substructure SMILES**: The core structure of this Bit (analysis target)\n"
                            f"- **Source Molecule SMILES**: Only for contextual reference\n"
                            f"Please analyze the potential impact of this feature on performance primarily based on the Substructure SMILES."
                        )
                    })
        
        messages.append({
            "role": "user", 
            "content": "Evidence presentation complete. Please integrate all [Structural Details], [Molecular Structure Images] (if any), [PDP Trends], and the aforementioned [Global Feature Importance Knowledge Base] to output the final JSON summary report.\n\n⚠️ Special Note: Please describe each Bit feature based on the actual chemical structure you observe in the molecular structure images, do not guess aimlessly. If you see specific functional groups, heterocycles, or other structural features, please describe them accurately."
        })

        print("First 1000 characters of summary messages input prompt:")
        # Safe print
        try:
            print(str(messages)[0:1000])
        except: pass

        print("\n" + "="*60)
        print("🤖 Requesting comprehensive reasoning from LLM...")
        print("="*60 + "\n")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Save Readable Log
            self._save_readable_log(messages, response)
            
            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                raise ValueError("LLM returned empty response")
            
            result_content = response.choices[0].message.content
            
            # Check for empty content
            if not result_content or result_content.strip() == "":
                raise ValueError("LLM returned empty content")
            
            print(f"📝 [Debug] Raw LLM Response (first 500 chars): {result_content[:500]}...")
            
            # Parse JSON
            try:
                result_json = json.loads(result_content)
            except json.JSONDecodeError:
                # Fallback: Try to extract JSON from markdown code block
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result_content)
                if json_match:
                    result_json = json.loads(json_match.group(1))
                else:
                    # Try to clean and parse
                    cleaned = result_content.strip().strip("`").strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    result_json = json.loads(cleaned)

            # Print full JSON report
            print("\n" + "="*60)
            print("📋 [SummaryAgent] Full Summary Report (JSON):")
            print("="*60)
            print(json.dumps(result_json, indent=4, ensure_ascii=False))
            print("="*60 + "\n")
            
            # Save report (Save to global run directory)
            report_path = os.path.join(get_run_dir(), "final_summary_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
            print(f"✅ Summary report saved to: {report_path}")

            return result_json

        except Exception as e:
            print(f"❌ Summary generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}