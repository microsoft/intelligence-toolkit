

def convert_to_sentences(df, skip):
    sentences = []
    cols = df.columns 
    for row in df.iter_rows(named=True):
        sentence = ''
        for field in cols:
            if field not in skip:
                val = str(row[field]).upper()
                if val == 'NAN':
                    val = ''
                sentence += field.upper() + ': ' + val + '; '
        sentences.append(sentence.strip())
    return sentences