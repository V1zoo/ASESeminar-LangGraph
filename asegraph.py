import json
import os
import subprocess
from typing import Annotated
from langchain.chat_models import init_chat_model
import requests
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, ToolCall
from langchain_core.messages.base import BaseMessage, message_to_dict
import pprint
from mytools import FileSearchTool

API_URL = "http://localhost:8081/task/index/"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
llm = init_chat_model("openai:gpt-4o-mini", base_url="http://188.245.32.59:4000")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    pending_tool_calls: List[ToolCall]
    problem: str
    repo_path: str
    bug_locator_done: bool
    code_editor_done: bool
    total_tokens: int
    rekursions: int

graph = StateGraph(State)

def router_after_tool_node(state):
    if state["rekursions"] > 31:
        return "end"
    if state.get("bug_locator_done", False):
        if state.get("code_editor_done", False):
            return "end"
        else:
            state["rekursions"] = 0
            return "code_editor"
    else:
        return "bug_locator"


def tool_node(state):
    print("ENTERING TOOL NODE")
    messages = state["messages"]
    tool_calls = state.get("pending_tool_calls", [])

    for call in tool_calls:
        if "function" in call:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
        else:
            name = call["name"]
            args = call["args"]

        tool_map = {
            "file_search": FileSearchTool(),
            "list_directory": ListDirectoryTool(),
            "read_file": ReadFileTool(),
            "write_file": WriteFileTool(),
        }

        tool = tool_map.get(name)
        if tool:
            print(f"\nInvoking {name} with args: {args}")
            try:
                result = tool.invoke(args)
            except Exception as e:
                result = f"The following error occured during the execution of {name}: {e}"
        else:
            result = f"Tool '{name}' not implemented."

        tool_msg = ToolMessage(
            tool_call_id=call["id"],
            content= result
        )
        print("\nConstructedToolMessage")
        pprint.pprint(tool_msg)
        messages.append(tool_msg)
    return {
        "messages": messages,
        "pending_tool_calls": [],
        "problem": state["problem"],
        "repo_path": state["repo_path"],
        "bug_locator_done": state["bug_locator_done"],
        "code_editor_done": state["code_editor_done"],
        "total_tokens": state["total_tokens"],
        "rekursions": (state["rekursions"] + 1)
    }

def bug_locator(state: State):
    print("ENTERING BUG LOCATOR")
    prompt = f"""You are an experienced AI developer. Locate the file in the directory {state["repo_path"]}
    that is responsible for the problem below. When you are done say 'DONE LOCATING'. Problem: {state["problem"]}"""
    llm_with_tools = llm.bind_tools([FileSearchTool(),ReadFileTool()]) #removed list directory tool
    messages = state["messages"]
    messages.append(HumanMessage(content=prompt))
    assert all(isinstance(m, BaseMessage) for m in messages)
    blocator_response = llm_with_tools.invoke(messages)
    pprint.pprint(message_to_dict(blocator_response))
    tokens_used = message_to_dict(blocator_response)["data"]["response_metadata"]["token_usage"]["total_tokens"]
    new_total_tokens = state["total_tokens"] + tokens_used
    if "DONE LOCATING" in blocator_response.content:
        bug_locator_done = True
    else:
        bug_locator_done = False
    print("\nBUG LOCATOR LLM RESPONSE")
    print(blocator_response)
    raw_tool_calls = getattr(blocator_response, "tool_calls", [])
    tool_calls = [
    ToolCall(
        name=tc["name"],
        args=tc["args"],
        id=tc["id"]
    ) for tc in raw_tool_calls
    ]
    tool_call_name = tool_calls[0]["name"] if tool_calls else "unknown_tool"
    ai_msg = AIMessage(content=blocator_response.content or f"[Calling tool: {tool_call_name}]", tool_calls=tool_calls)
    messages.append(ai_msg)
    return {"messages": messages,
            "pending_tool_calls": blocator_response.tool_calls or [], # type: ignore
             "problem": state["problem"], "repo_path": state["repo_path"], "bug_locator_done": bug_locator_done,
             "code_editor_done": state["code_editor_done"], "total_tokens": new_total_tokens, "rekursions": state["rekursions"]}

def code_editor(state: State):
    print(f"ENTERING CODE EDITOR WITH RECURSIONS {state["rekursions"]}")
    prompt = f"""You're a professional AI developer currently assigned to bug-fixing.
    You are working in the directory: {state["repo_path"]}. This is a Git repository.
    Implement the fix for the problem described below.
    The Bug Locator has already figured out the origin of the problem described below.
    All code changes must be saved to the files, so they appear in `git diff`.
    Ensure that the changes are written to the files by using the file writing tool WriteFileTool() and passing the fixed code as an argument.
    You are allowed to overwrite files in this directory.
    Ensure that the fix is minimal and only changes whats necessary to resolve the problem.
    Important: When you have finished, say 'DONE EDITING'.
    Problem description: 
    {state["problem"]}"""
    messages = state["messages"] + [{"role": "developer", "content": prompt}]
    llm_with_tools = llm.bind_tools([FileSearchTool(), ReadFileTool(), WriteFileTool()])
    ceditor_response = llm_with_tools.invoke(messages)
    tokens_used = message_to_dict(ceditor_response)["data"]["response_metadata"]["token_usage"]["total_tokens"]
    new_total_tokens = state["total_tokens"] + tokens_used
    if "DONE EDITING" in ceditor_response.content:
        code_editor_done = True
    else:
        code_editor_done = False
    if state["rekursions"] > 29:
        code_editor_done = True
    print("\nCODE EDITOR LLM RESPONSE")
    pprint.pprint(ceditor_response)
    raw_tool_calls = getattr(ceditor_response, "tool_calls", [])
    tool_calls = [
    ToolCall(
        name=tc["name"],
        args=tc["args"],
        id=tc["id"]
    ) for tc in raw_tool_calls
    ]
    tool_call_name = tool_calls[0]["name"] if tool_calls else "unknown_tool"
    ai_msg = AIMessage(content=ceditor_response.content or f"[Calling tool: {tool_call_name}]", tool_calls=tool_calls)
    messages.append(ai_msg)
    return {
        "messages": messages,
        "pending_tool_calls": ceditor_response.tool_calls or [], # type: ignore
        "repo_path": state["repo_path"],
        "problem": state["problem"],
        "bug_locator_done": state["bug_locator_done"],
        "code_editor_done": code_editor_done,
        "total_tokens": new_total_tokens,
        "rekursions": state["rekursions"]
    }

graph.add_node("bug_locator", bug_locator)
graph.add_node("code_editor", code_editor)
graph.add_node("tool_node", tool_node)
graph.add_node("router_after_tool_node", router_after_tool_node)
graph.set_entry_point("bug_locator")

graph.add_edge("bug_locator", "tool_node")
graph.add_edge("code_editor", "tool_node")

graph.add_conditional_edges("tool_node", router_after_tool_node, {
    "bug_locator": "bug_locator",
    "code_editor": "code_editor",
    "end": END
})

app = graph.compile()

#range(30,31)
for index in range(1,31):
        api_url = f"{API_URL}{index}"
        print(f"Fetching test case {index} from {api_url}")
        start_dir = os.getcwd()     #get current working directory
        repo_dir = os.path.join(start_dir, f"repos\\repo_{index}")
        LOG_FILE = start_dir + "\\LGLog.txt"
        print(f"{start_dir}      {repo_dir}")
        try:
            responseJSON = requests.get(f"{api_url}").json()
            taskNumber = responseJSON['taskNumber']
            instance_id = responseJSON['instance_id']
            prompt = responseJSON["Problem_statement"]
            git_clone = responseJSON["git_clone"]
            fail_to_pass = json.loads(responseJSON.get("FAIL_TO_PASS","[]"))
            pass_to_pass = json.loads(responseJSON.get("PASS_TO_PASS","[]"))
            print("\n-<Task Number>-------------------------------\n")
            print(taskNumber)
            print("\n-<Instance ID>-------------------------------\n")
            print(instance_id)
            print("\n-<Problem Statement>-------------------------\n")
            print(prompt)
            print("\n-<Git Clone>---------------------------------\n")
            print(git_clone)
            print("\n-<Fail to Pass>------------------------------\n")
            print(fail_to_pass)
            print("\n-<Pass to Pass>------------------------------\n")
            print(pass_to_pass)

            # Extract repo URL and commit hash
            parts = git_clone.split("&&")
            clone_part = parts[0].strip()
            checkout_part = parts[-1].strip() if len(parts) > 1 else None
            repo_url = clone_part.split()[2]
            print(f"Cloning repository {repo_url} into {repo_dir}...")
            env = os.environ.copy()
            env["GIT_TERMINAL_PROMT"] = "0"
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True, env= env)

            if checkout_part:
                commit_hash = checkout_part.split()[-1]
                print(f"Checking out commit: {commit_hash}")
                subprocess.run(["git", "checkout", commit_hash], cwd=repo_dir, check=True, env=env)

        except:
            raise Exception("Invalid response")

        inputs = {
            'index': index,
            'prompt': prompt # type: ignore
        }

        try:
                       
            initial_state: State = {
                "messages":[],
                "pending_tool_calls": [],
                "problem": prompt,
                "repo_path": f"repos\\repo_{index}",
                "bug_locator_done": False,
                "code_editor_done": False,
                "total_tokens": 0,
                "rekursions": 0
            }

            result = app.invoke(State(initial_state), config={"recursion_limit": 64})

            tokens = result["total_tokens"]

            print(f"Calling SWE-Bench REST service with repo: {repo_dir}")
            test_payload = {
                "instance_id": instance_id, # type: ignore
                "repoDir" : f"/repos/repo_{index}",  # mount with docker?
                "FAIL_TO_PASS" : fail_to_pass, # type: ignore
                "PASS_TO_PASS" : pass_to_pass, # type: ignore
            }
            print(test_payload)
            result = requests.post("http://localhost:8082/test", json=test_payload)
            result.raise_for_status()
            print(f"benchresponse: {result.content}")
            if len(result.json()) == 1:
                os.chdir(start_dir)
                with open(LOG_FILE, "a", encoding="utf-8") as log:
                    log.write(f"\n---< TESTCASE {index} >------------")
                    log.write(f"\nFailed to make any changes to the repository or")
                    log.write("\nencountered errors during evaluation")
                    log.write(f"\nTotal Tokens used: {tokens}")
                print(f"Test case {index} unchanged and logged")
            else:
                result_raw = result.json().get("harnessOutput", "{}")
                result_json = json.loads(result_raw)
                if not result_json:
                    print(f"BenchResponseError: {result.json().get("error")}")
                    raise ValueError(f"No data in harnessOutput - possible evaluation error\nTotal Tokens used: {tokens}")
                print(result_json)
                instance_id = next(iter(result_json))
                tests_status = result_json[instance_id]["tests_status"]
                fail_pass_results = tests_status["FAIL_TO_PASS"]
                fail_pass_passed = len(fail_pass_results["success"])
                fail_pass_total = fail_pass_passed + len(fail_pass_results["failure"])
                pass_pass_results = tests_status["PASS_TO_PASS"]
                pass_pass_passed = len(pass_pass_results["success"])
                pass_pass_total = pass_pass_passed + len(pass_pass_results["failure"])

                # log results
                os.chdir(start_dir)
                with open(LOG_FILE, "a", encoding="utf-8") as log:
                    log.write(f"\n---< TESTCASE {index} >------------")
                    log.write(f"\nFAIL_TO_PASS passed: {fail_pass_passed}/{fail_pass_total}")
                    log.write(f"\nPASS_TO_PASS passed: {pass_pass_passed}/{pass_pass_total}")
                    log.write(f"\nTotal Tokens used: {tokens}")
                print(f"Test case {index} completed and logged")
        except Exception as e:
            os.chdir(start_dir)
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"\n---< TESTCASE {index} >------------")
                log.write(f"\nError: {e}")
            print(f"Error in test case {index}: {e}")
            raise e