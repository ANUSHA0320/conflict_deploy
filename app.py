from flask import Flask, render_template, request
import gitlab
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# GitLab Auth
GITLAB_URL = "https://gitlab.com"
PRIVATE_TOKEN = os.getenv("PRIVATE_TOKEN", "glpat-i55dAwxXT9v9r2yRzKQc")  # or better: use GitLab CI secret variable

gl = gitlab.Gitlab(GITLAB_URL, private_token=PRIVATE_TOKEN)

# Load model
model_path = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prompt builder
def build_conflict_prompt(conflicted_code: str) -> str:
    return f"""You are an expert software developer well versed with Knowledge of logic behind the code and you are able to identify the reason behind git merge conflict and you are an expert in giving the correct code.
Here is a merge conflict block from a file:

{conflicted_code}

Tasks:
1. Explain why the conflict occurred.
2. Highlight the differing lines between both branches.
3. give the correct code to achieve the task based on your knowledge

### Response:
"""

# Bot response
def get_bot_response(prompt, max_tokens=600):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response.split("### Response:")[-1].strip()

# Git file fetchers
def fetch_file_content(project, branch, file_path):
    try:
        f = project.files.get(file_path=file_path, ref=branch)
        return f.decode().decode('utf-8')
    except Exception as e:
        return f"❌ Error: {e}"

def create_merge_conflict_block(source_content, target_content, source_branch, target_branch):
    return f"""<<<<<<< {target_branch}
{target_content}
=======
{source_content}
>>>>>>> {source_branch}"""


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        project_path = request.form['project_path'].strip()
        try:
            project = gl.projects.get(project_path)
            # Fetch all merge requests (set 'all=True' to fetch all pages)
            merge_requests = project.mergerequests.list(state='opened', all=True)

            # Filter the merge requests that have conflicts
            conflicted_mrs = [mr for mr in merge_requests if mr.has_conflicts]

            if not conflicted_mrs:
                result = "✅ No merge requests with conflicts found."
            else:
                output = []
                for mr in conflicted_mrs:
                    src = mr.source_branch
                    tgt = mr.target_branch
                    changes = mr.changes()

                    # Iterate over all changes in the merge request
                    for change in changes['changes']:
                        # Check if the file is not a new or deleted file
                        if not change.get('new_file') and not change.get('deleted_file'):
                            fp = change['new_path']
                            src_content = fetch_file_content(project, src, fp)
                            tgt_content = fetch_file_content(project, tgt, fp)

                            if "❌" in src_content or "❌" in tgt_content:
                                continue

                            # Create a conflict block for each file conflict
                            block = create_merge_conflict_block(src_content, tgt_content, src, tgt)
                            prompt = build_conflict_prompt(block)
                            analysis = get_bot_response(prompt)

                            # Add the result to output
                            output.append((fp, analysis))

                # If output is empty, no conflicts were detected in any of the files
                if not output:
                    result = "✅ No conflicts found in the merge request files."
                else:
                    result = output

        except Exception as e:
            result = f"❌ Error: {e}"

    return render_template("index.html", result=result)


if __name__ == '__main__':
    app.run(debug=True)