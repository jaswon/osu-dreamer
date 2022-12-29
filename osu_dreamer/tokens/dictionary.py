TO_IDX = { x:i for i,x in enumerate([
    'START_TIME',
    'END_TIME',
    'POSITION',
    'NEW_COMBO',
    'SPINNER',
    'CIRCLE',
    'SLIDER',
    'LINE',
    'CUBIC',
    'SLIDE',
    'END',
    'EOS', # end of sequence
    'BOS', # beginning of sequence
    'PAD', # padding token
])}

START_TIME = TO_IDX['START_TIME']
END_TIME = TO_IDX['END_TIME']
POSITION = TO_IDX['POSITION']
END = TO_IDX['END']
EOS = TO_IDX['EOS']
BOS = TO_IDX['BOS']
PAD = TO_IDX['PAD']

FROM_IDX = { i:x for x,i in TO_IDX.items() }

VOCAB_SIZE = len(TO_IDX)
