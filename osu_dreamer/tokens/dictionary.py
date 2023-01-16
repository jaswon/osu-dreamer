TO_IDX = { x:i for i,x in enumerate([
    'TIME',
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
    *( f'X{x:+d}' for x in range(-256,512+256) ),
    *( f'Y{y:+d}' for y in range(-192,384+192) ),
])}

TIME = TO_IDX['TIME']
END = TO_IDX['END']
EOS = TO_IDX['EOS']
BOS = TO_IDX['BOS']
PAD = TO_IDX['PAD']

FROM_IDX = { i:x for x,i in TO_IDX.items() }

VOCAB_SIZE = len(TO_IDX)