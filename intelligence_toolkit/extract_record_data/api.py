import intelligence_toolkit.extract_record_data.data_extractor as data_extractor
import intelligence_toolkit.generate_mock_data.data_generator as data_generator
from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration
import pandas as pd


class ExtractRecordData:
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

    async def extract_record_data(
        self,
        input_texts: list[str],
        generation_guidance: str = "",
        df_update_callback=None,
        callback_batch=None,
    ):
        """
        Extracts structured data records from input texts according to the JSON schema

        Args:
            input_texts (list[str]): The list of input texts to extract data from
            generation_guidance (str): Optional guidance to provide to the model
            df_update_callback (function): A callback function to update the dataframe
            callback_batch (function): A callback function to update the batch
        """
        self.json_object, self.array_dfs = await data_extractor.extract_record_data(
            ai_configuration=self.ai_configuration,
            input_texts=input_texts,
            data_schema=self.json_schema,
            record_arrays=self.record_arrays,
            generation_guidance=generation_guidance,
            df_update_callback=df_update_callback,
            callback_batch=callback_batch,
        )
