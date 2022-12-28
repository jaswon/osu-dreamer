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
    'BSS', # before sequence start
    *( f'X{x:d}' for x in range(512) ),
    *( f'Y{y:d}' for y in range(384) ),
])}

TIME = TO_IDX['TIME']
EOS = TO_IDX['EOS']
BSS = TO_IDX['BSS']

FROM_IDX = { i:x for x,i in TO_IDX.items() }

VOCAB_SIZE = len(TO_IDX)