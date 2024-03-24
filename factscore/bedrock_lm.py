from factscore.lm import LM
import boto3
from botocore.exceptions import ClientError
import json

import sys
import time
import os
import numpy as np
import logging

class BedRockAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        # init bedrock runtime client
        self.client = boto3.client(
            service_name = "bedrock-runtime",
            region_name = "us-east-1"
        )
        self.model_name = model_name
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        pass

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "Titan":
            # Construct the prompt send to ChatGPT
            message = prompt
            # Call API
            response = self._call_titan(message)
            # Get the output from the response
            print("response : {}".format(response))
            output = response["results"][0]["outputText"]
            return output, response
        elif self.model_name == "Llama2":
            # Call API
            response = self._call_llama2(prompt)
            # Get the output from the response
            print(response)
            output = response["choices"][0]["text"]
            return output, response
        else:
            raise NotImplementedError()


    def _call_titan(self, message):
        # invoke titan with the text prompt
        model_id = "amazon.titan-text-lite-v1"

        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "inputText": message
                    }
                ),
            )

            # process and print the response
            result = json.loads(response.get("body").read())
            input_tokens = result["inputTextTokenCount"]
            output_tokens = 0
            output_list = result.get("results", [])

            for output in output_list:
                output_tokens += output["tokenCount"]

            #print("Invocation details:")
            #print(f"- The input length is {input_tokens} tokens.")
            #print(f"- The output length is {output_tokens} tokens.")

            #print(f"- The model returned {len(output_list)} response(s):")
            for output in output_list:
                print("output : {}".format(output["outputText"]))
            print(result)
            return result

        except ClientError as err:
            print(
                "Couldn't invoke Titan Embeddings G1 - Text. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise
    
    def _call_llama2(self, message):
        # invoke llama2 with the text prompt
        model_id = "meta.llama2-13b-chat-v1"

        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "prompt": message,
                        "max_gen_len": 512,
                        "temperature": 0.5,
                        "top_p": 0.9,
                    }
                ),
            )

            # process and print the response
            result = json.loads(response.get("body").read())
          
            input_tokens = result["prompt_token_count"]
            output_tokens = result["generation_token_count"]
            output = result["generation"]

            print("Invocation details:")
            print(f"- The input length is {input_tokens} tokens.")
            print(f"- The output length is {output_tokens} tokens.")

            print(f"- The model returned 1 response(s):")
            print(output)

            return result

        except ClientError as err:
            print(
                "Couldn't invoke Llama 2 Chat 13B. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise