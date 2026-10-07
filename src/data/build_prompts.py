import argparse
import os
import pandas as pd
from utils import load_data, save_data

BASE_URL = "https://openrouter.ai/api/v1"

class PromptBuilder:
    """Builds input prompts from prompt templates and data for querying models."""

    def __init__(
            self,
            user_prompt_path: str,
            system_prompt_path: str | None = None,
            data_path: str | None = None,
        ):
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.data_path = data_path

    def set_system_prompts(self, system_prompt_path: str | None):
        """Reads system prompt templates from a file."""

        self.system_prompts = []
        system_prompt_path = system_prompt_path or self.system_prompt_path
        if system_prompt_path is not None and os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r") as f:
                self.system_prompts.append(f.read())
        else:
            raise ValueError(f"System prompt path '{system_prompt_path}' does not exist.")

    def set_user_prompts(self, user_prompt_path: str | None, data_path: str | None):
        """Reads user prompt templates and data to generate user prompts."""

        self.user_prompts = []
        user_prompt_path = user_prompt_path or self.user_prompt_path
        data_path = data_path or self.data_path
        if user_prompt_path is not None and os.path.exists(user_prompt_path):
            with open(user_prompt_path, "r") as f:
                user_prompt_template = f.read()
                data = pd.read_csv(data_path) if data_path is not None else pd.DataFrame()
                user_prompts = [
                    {
                        "word": word, 
                        "prompt": user_prompt_template.format(WORD=word)
                    } for word in data.word_string.tolist()
                ]
                self.user_prompts.extend(user_prompts)
        else:
            raise ValueError(f"User prompt path '{user_prompt_path}' does not exist.")

    def build_prompt_dataset(
            self, 
            system_prompt_path: str | None = None, 
            user_prompt_path: str | None = None, 
            data_path: str | None = None, 
            save_path: str | None = None
        ):
        """Builds a dataset of prompts by combining system and user prompts."""

        self.set_system_prompts(system_prompt_path)
        self.set_user_prompts(user_prompt_path, data_path)

        dataset = []
        for system_prompt in self.system_prompts:
            for user_prompt in self.user_prompts:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt["prompt"]
                    }
                ]
                dataset.append(
                    {
                        "word": user_prompt["word"],
                        "messages": messages
                    }
                )
        df = pd.DataFrame.from_records(dataset)
        return df


        
def main():
    parser = argparse.ArgumentParser(description="Build input prompts for querying models.")
    parser.add_argument("--prompt-template-path", default=None, help="optional path to a prompt template")
    parser.add_argument("--data-path", default=None, help="Dataset with values to use the prompt template on")
    parser.add_argument("--save-path", default=None, help="Path to save the generated prompts")
    args = parser.parse_args()

    if args.prompt_template_path:
        sys_path = os.path.join(args.prompt_template_path, "system.txt")
        user_path = os.path.join(args.prompt_template_path, "user.txt")
        
    else:
        raise ValueError("No prompt template provided.")

    pb = PromptBuilder(
        user_prompt_path=user_path,
        system_prompt_path=sys_path,
        data_path=args.data_path,
    )
    df = pb.build_prompt_dataset()
    df = pd.DataFrame(df)
    if args.save_path is not None: 
        save_data(df, args.save_path)

if __name__ == "__main__":
    main()
