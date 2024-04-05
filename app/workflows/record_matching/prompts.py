system_prompt = """\
Goal: evaluate the overall RELATEDNESS of the records in each record group provided on a scale of 0-10, where 0 is definitively different entities and 10 is definitivly the same entity or entity group (e.g., branches of a company).

Factors indicating unrelatedness: multiple fields having values that are different across grouped records, have no similarity, and are unrelated in the real-world.

Factors indicating relatedness: multiple fields having values that are the same or similar across multiple grouped records, and are related in the real-world.

Factors that should be ignored: inconsistent spelling, formatting, and missing values.

Factors that should be considered in the event of similar names: the more additional fields that are the same, the more likely the records are related.

If names are in a language other than English, consider whether the English translations are generic descriptive terms (less likely to be related) or distinctive (more likely to be related).

Output the rows of a CSV file containing the Group ID, Relatedness, and Explanation. Keep explanations short and simple.

Do not output ``` or the column headers - start directly with the row values and separate each row with a newline. Output Group ID and Relatedness directly, but wrap explanations in "".

Additional instructions:

{instructions}

=== TASK ===

Group data:

{data}
"""