#### Модифицированный код лежит в `nle/agent/agent.py`

### Добавлены аргументы:

`--tf_layers` - количество слоев трансформера

`--tf_nheads` - количество attention голов

`--dim_feedforward` - размерность feedforward модели трансформера

`--tf_maxlen` - максимальная длина последовательности, обрабатываемой трансформером. При значении 0
последовательности обрабатываются без ограничения по длине

### Пример запуска: </br>
`python agent.py --use_lstm --num_actors 80 --batch_size 32
--unroll_length 80 --learning_rate 0.0001
--entropy_cost 0.0001  --total_steps 1000000000
--savedir ./test/ --tf_layers 2 --tf_nheads 4
--dim_feedforward 512 --tf_maxlen 200`
