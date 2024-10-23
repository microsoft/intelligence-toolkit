import toolkit.generate_mock_data.data_generator as data_generator
import toolkit.generate_mock_data.text_generator as text_generator

class GenerateMockData:
    def __init__(self):
        self.json_schema = {}
        self.record_arrays = []
        self.json_object = {}
        self.array_dfs = {}

    def set_schema(
        self,
        json_schema: dict
    ):
        self.json_schema = json_schema
        self.record_arrays = data_generator.extract_array_fields(json_schema)

    def set_ai_configuration(
        self,
        ai_configuration
    ):
        self.ai_configuration = ai_configuration

    async def generate_data_records(
        self,
        num_records_overall,
        records_per_batch,
        duplicate_records_per_batch,
        related_records_per_batch,
        generation_guidance="",
        temperature=0.5,
        df_update_callback=None,
        callback_batch=None,
        parallel_batches=0
    ):
       
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
            parallel_batches=parallel_batches
        )

    async def generate_text_data(
        self,
        df,
        generation_guidance="",
        temperature=0.5,
        df_update_callback=None
    ):
        input_texts = []
        for _, row in df.iterrows():
            input_texts.append(row.to_json())
        self.text_list, self.text_df = await text_generator.generate_text_data(
            self.ai_configuration,
            generation_guidance,
            input_texts,
            temperature,
            df_update_callback
        )