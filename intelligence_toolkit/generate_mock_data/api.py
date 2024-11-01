import intelligence_toolkit.generate_mock_data.data_generator as data_generator
import intelligence_toolkit.generate_mock_data.text_generator as text_generator
from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration
import pandas as pd


class GenerateMockData:
    def __init__(self):
        self.json_schema = {}
        self.record_arrays = []
        self.json_object = {}
        self.array_dfs = {}

    def set_schema(self, json_schema: dict):
        self.json_schema = json_schema
        self.record_arrays: list[list[str]] = data_generator.extract_array_fields(
            json_schema
        )

    def set_ai_configuration(self, ai_configuration: OpenAIConfiguration):
        self.ai_configuration = ai_configuration

    async def generate_data_records(
        self,
        num_records_overall: int,
        records_per_batch: int,
        duplicate_records_per_batch: int,
        related_records_per_batch: int,
        generation_guidance: str = "",
        temperature: float = 0.5,
        df_update_callback=None,
        callback_batch=None,
        parallel_batches: int = 0,
    ):
        """
        Generates structured data records according to the JSON schema

        Args:
            num_records_overall (int): The total number of records to generate
            records_per_batch (int): The number of records to generate per batch
            duplicate_records_per_batch (int): The number of duplicate records to generate per batch
            related_records_per_batch (int): The number of related records to generate per batch
            generation_guidance (str): Optional guidance to provide to the model
            temperature (float): The temperature to use when generating data
            df_update_callback (function): A callback function to update the dataframe
            callback_batch (function): A callback function to update the batch
            parallel_batches (int): The number of parallel batches to generate
        """
        self.json_object, self.array_dfs = await data_generator.generate_data(
            ai_configuration=self.ai_configuration,
            generation_guidance=generation_guidance,
            data_schema=self.json_schema,
            num_records_overall=num_records_overall,
            records_per_batch=records_per_batch,
            duplicate_records_per_batch=duplicate_records_per_batch,
            related_records_per_batch=related_records_per_batch,
            temperature=temperature,
            df_update_callback=df_update_callback,
            callback_batch=callback_batch,
            parallel_batches=parallel_batches,
        )

    async def generate_text_data(
        self,
        df: pd.DataFrame,
        generation_guidance: str = "",
        temperature: float = 0.5,
        df_update_callback=None,
    ):
        """
        Generates text data based on the input dataframe

        Args:
            df (pandas.DataFrame): The input dataframe
            generation_guidance (str): Optional guidance to provide to the model
            temperature (float): The temperature to use when generating data
            df_update_callback (function): A callback function to update the dataframe
        """
        input_texts = []
        for _, row in df.iterrows():
            input_texts.append(row.to_json())
        self.text_list, self.text_df = await text_generator.generate_text_data(
            ai_configuration=self.ai_configuration,
            input_texts=input_texts,
            generation_guidance=generation_guidance,
            temperature=temperature,
            df_update_callback=df_update_callback,
        )