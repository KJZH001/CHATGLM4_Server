"""
This script implements an API for the ChatGLM3-6B model, and adapted to GLM4-9b
formatted similarly to OpenAI's API (https://platform.openai.com/docs/api-reference/chat).
It's designed to be run as a web server using FastAPI and uvicorn,
making the ChatGLM3-6B model accessible through OpenAI Client.

Key Components and Features:
- Model and Tokenizer Setup: Configures the model and tokenizer paths and loads them.
- FastAPI Configuration: Sets up a FastAPI application with CORS middleware for handling cross-origin requests.
- API Endpoints:
  - "/v1/models": Lists the available models, specifically ChatGLM3-6B.
  - "/v1/chat/completions": Processes chat completion requests with options for streaming and regular responses.
  - "/v1/embeddings": Processes Embedding request of a list of text inputs.
- Token Limit Caution: In the OpenAI API, 'max_tokens' is equivalent to HuggingFace's 'max_new_tokens', not 'max_length'.
For instance, setting 'max_tokens' to 8192 for a 6b model would result in an error due to the model's inability to output
that many tokens after accounting for the history and prompt tokens.
- Stream Handling and Custom Functions: Manages streaming responses and custom function calls within chat responses.
- Pydantic Models: Defines structured models for requests and responses, enhancing API documentation and type safety.
- Main Execution: Initializes the model and tokenizer, and starts the FastAPI app on the designated host and port.

Note:
    This script doesn't include the setup for special tokens or multi-GPU support by default.
    Users need to configure their special tokens and can enable multi-GPU support as per the provided instructions.
    Embedding Models only support in One GPU.

    Running this script requires 14-15GB of GPU memory. 2 GB for the embedding model and 12-13 GB for the FP16 ChatGLM3 LLM.


"""

import os
import time
import tiktoken
import torch
import uvicorn
import json
import argparse
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from utils import process_response, generate_chatglm3, generate_stream_chatglm3
from sentence_transformers import SentenceTransformer
from tools.schema import tool_class, tool_def, tool_param_start_with
from sse_starlette.sse import EventSourceResponse

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

# set LLM path
# MODEL_PATH = os.environ.get('MODEL_PATH', 'E:\Project\GLM4\glm-4-9b-chat')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
# set Embedding Model path
# EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'E:\Project\GLM3\[Model]bge-large-zh-v1.5')

# Set LLM path via command-line arguments
parser = argparse.ArgumentParser(description="Set model and embedding paths.")
parser.add_argument('--model_path', type=str, default='E:\Project\GLM4\glm-4-9b-chat', help='Path to the LLM model')
parser.add_argument('--embedding_path', type=str, default='E:\Project\GLM3\[Model]bge-large-zh-v1.5', help='Path to the embedding model')
args = parser.parse_args()

MODEL_PATH = args.model_path
TOKENIZER_PATH = MODEL_PATH
EMBEDDING_PATH = args.embedding_path

print("======== Model and Tokenizer Paths ========")
print(f"Model Path: {MODEL_PATH}")
print(f"Tokenizer Path: {TOKENIZER_PATH}")
print(f"Embedding Path: {EMBEDDING_PATH}")
print("===========================================")



@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


## for Embedding
class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage


# for ChatCompletionRequest

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1
    tool_choice: Optional[Union[str, dict]] = "None"
    agent: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if isinstance(request.input, str):
        embeddings = [embedding_model.encode(request.input)]
    else:
        embeddings = [embedding_model.encode(text) for text in request.input]
    embeddings = [embedding.tolist() for embedding in embeddings]

    def num_tokens_from_string(string: str) -> int:
        """
        Returns the number of tokens in a text string.
        use cl100k_base tokenizer
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    response = {
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": CompletionUsage(
            prompt_tokens=sum(len(text.split()) for text in request.input),
            completion_tokens=0,
            total_tokens=sum(num_tokens_from_string(text) for text in request.input),
        )
    }
    return response


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(
        id="chatglm3-6b"
    )
    return ModelList(
        data=[model_card]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    # print(request.tools)
    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        tool_choice=request.tool_choice,
        agent=request.agent

    )
    logger.debug(f"==== request ====\n{gen_params}")
    gen_params["tools"] = gen_params['tools'] if gen_params["agent"] or gen_params['tools'] else []

    if request.stream:

        # Use the stream mode to read the first few characters, if it is not a function call, direct stram output
        predict_stream_generator = predict_stream(request.model, gen_params)
        output = next(predict_stream_generator)
        if not contains_custom_function(output, gen_params["tools"]):
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")

        # Obtain the result directly at one time and determine whether tools needs to be called.
        logger.debug(f"First result output：\n{output}")

        function_call = None
        if output and request.tools:
            try:
                function_call = process_response(output, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        # CallFunction
        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

            """
            In this demo, we did not register any tools.
            You can use the tools that have been implemented in our `tools_using_demo` and implement your own streaming tool implementation here.
            Similar to the following method:
            """
            if tool_param_start_with in output:
                tool = tool_class.get(function_call.name)
                if tool:
                    tool_param = json.loads(function_call.arguments).get("symbol")
                    if tool().parameter_validation(tool_param):
                        observation = str(tool().run(tool_param))
                        tool_response = observation
                    else:
                        tool_response = "Tool parameter values error, please tell the user about this situation."
                else:
                    tool_response = "No available tools found, please tell the user about this situation."
            else:
                tool_response = "Tool parameter content error, please tell the user about this situation."

            if not gen_params.get("messages"):
                gen_params["messages"] = []

            gen_params["messages"].append(ChatMessage(
                role="assistant",
                content=output,
            ))
            gen_params["messages"].append(ChatMessage(
                role="function",
                name=function_call.name,
                content=tool_response,
            ))

            # Streaming output of results after function calls
            generate = predict(request.model, gen_params)
            return EventSourceResponse(generate, media_type="text/event-stream")

        else:
            # Handled to avoid exceptions in the above parsing function process.
            generate = parse_output_text(request.model, output)
            return EventSourceResponse(generate, media_type="text/event-stream")

    # Here is the handling of stream = False
    response = generate_chatglm3(model, tokenizer, gen_params)

    # Remove the first newline character
    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()

    usage = UsageInfo()
    function_call, finish_reason = None, "stop"
    if request.tools:
        try:
            function_call = process_response(response["text"], use_tool=True)
        except:
            logger.warning("Failed to parse tool call, maybe the response is not a tool call or have been answered.")

    if isinstance(function_call, dict):
        finish_reason = "function_call"
        function_call = FunctionCallResponse(**function_call)

    message = ChatMessage(
        role="assistant",
        content=response["text"],
        function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
    )

    logger.debug(f"==== message ====\n{message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        id="",  # for open_source model, id is empty
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )


async def predict(model_id: str, params: dict):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        finish_reason = new_response["finish_reason"]
        if len(delta_text) == 0 and finish_reason != "function_call":
            continue

        function_call = None
        if finish_reason == "function_call":
            try:
                function_call = process_response(decoded_unicode, use_tool=True)
            except:
                logger.warning(
                    "Failed to parse tool call, maybe the response is not a tool call or have been answered.")

        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id="",
            choices=[choice_data],
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        id="",
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def predict_stream(model_id, gen_params):
    """
    The function call is compatible with stream mode output.

    The first seven characters are determined.
    If not a function call, the stream output is directly generated.
    Otherwise, the complete character content of the function call is returned.

    :param model_id:
    :param gen_params:
    :return:
    """
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    for new_response in generate_stream_chatglm3(model, tokenizer, gen_params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode

        # When it is not a function call and the character length is> 7,
        # try to judge whether it is a function call according to the special function prefix
        if not is_function_call and len(output) > 7:

            # Determine whether a function is called
            is_function_call = contains_custom_function(output, gen_params["tools"])
            if is_function_call:
                continue

            # Non-function call, direct stream output
            finish_reason = new_response["finish_reason"]

            # Send an empty string first to avoid truncation by subsequent next() operations.
            if not has_send_first_chunk:
                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id="",
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk"
                )
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

            send_msg = delta_text if has_send_first_chunk else output
            has_send_first_chunk = True
            message = DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id="",
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk"
            )
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    if is_function_call:
        yield output
    else:
        #parse_openai_v1_12_0(output)
        #logger.warning("==== 548 DONE  ==== \n")

        """
        If the input content is related to function calls, then it will be converted to openai v1.12.0 format.
        
        """
        def make_bin(a,all_ = False):
            # a是全文或内容部分； 文本转字节流
            # a默认是内容部分
            if all_:
                x2x_json = a
            else:
                x2x_json = {"choices":[{"delta":{"content":a}}]}
            x3x_str = json.dumps(x2x_json,ensure_ascii=False)
            x4x_str = f"data: {x3x_str}\n\n"
            x5x_byte = x4x_str.encode('utf-8')
            return x5x_byte
                    
        if "tool_call(" in output and "```" in output and ")" in output:
                #logger.warning(f"==== tool_call  ==== \n 需要工具\n全文：{output}")
                
                def tool_call(**kwargs):
                    return kwargs
                
                output = output.splitlines()
                function_name,args = output[0].strip(),eval(output[2].strip())
                args = json.dumps(args, ensure_ascii=False)
                #logger.warning(f"==== tool_function  ==== \n{function_name}")
                #logger.warning(f"==== tool_function_args  ==== \n{args}")
                
                # 发函数名
                xx_1 = {"id":"chatcmpl-8x6K1Jxc12Q6cpKNI0QHfsLZadkR2","object":"chat.completion.chunk","created":1709096345,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_86156a94a0","choices":[{"index":0,"delta":{"role":"assistant","content":"null","tool_calls":[{"index":0,"id":"call_F4zBrDWWXGhAmqW2iK3U92X2","type":"function","function":{"name":function_name,"arguments":""}}]},"logprobs":"null","finish_reason":"null"}]}
                yield make_bin(xx_1,True)
                
                # 发参数部分
                for i in args:
                    xx_2 = {"id":"chatcmpl-8x6K1Jxc12Q6cpKNI0QHfsLZadkR2","object":"chat.completion.chunk","created":1709096345,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_86156a94a0","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":i}}]},"logprobs":"null","finish_reason":"null"}]}
                    yield make_bin(xx_2,True)
                # 发函数调用结束
                xx_3 = {"id":"chatcmpl-8x6K1Jxc12Q6cpKNI0QHfsLZadkR2","object":"chat.completion.chunk","created":1709096345,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_86156a94a0","choices":[{"index":0,"delta":{},"logprobs":"null","finish_reason":"tool_calls"}]}
                yield make_bin(xx_3,True)
                
                # 发完毕
                yield b'data: [DONE]\n\n'
                
        else:
            yield make_bin('')
            
        yield '[DONE]'


async def parse_output_text(model_id: str, value: str):
    """
    Directly output the text content of value

    :param model_id:
    :param value:
    :return:
    """
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=value),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def contains_custom_function(value: str, tools: list) -> bool:
    """
    Determine whether 'function_call' according to a special function prefix.
    [Note] This is not a rigorous judgment method, only for reference.

    :param value:
    :param tools:
    :return:
    """
    for tool in tools:
        if value and tool["name"] in value:
            return True


if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)  # 或 load_in_8bit=True
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto", quantization_config=quantization_config)
    logger.add('apis.log',rotation='5 MB',enqueue=True,serialize=False,encoding='utf-8',retention='10 days')

    # load Embedding
    embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cuda")
    uvicorn.run(app, host='0.0.0.0', port=31256, workers=1)
