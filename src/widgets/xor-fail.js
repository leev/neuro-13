makeSingleTrainer({
  data: [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],0]],
  labels: ['left','right'],
  predIds: 'x-predictions',
  statIds: { epoch: 'x-epoch', err: 'x-err' },
  btnIds: { step: 'x-step', fast: 'x-fast', reset: 'x-reset' },
  modelId: 'x-model',
  markUnsure: true,
});


makeSeparator('sep-xor',
  [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],0]],
  ['left','right'],
  { start: [-1, 1, 0.5] });

