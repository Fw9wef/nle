#### Модифицированный код лежит в `nle/agent/agent.py` и `nle/agent/gtrxl_agent.py`
Для использования трансформера при запуске необходимо
указать ключ `--use_lstm`. В `agent.py` реализован gated трансформер с абсолютным позиционированием
относительно начала эпизода. В `gtrxl_agent.py` реализован gated transformer-XL с относительным позиционированием
(как в статье https://arxiv.org/abs/1910.06764). Также в `gtrxl_agent.py` реализован
быстрый форвард MHA (без пересчета вычисленных на прошлом шаге значений)

### Добавлены аргументы:

`--tf_layers` - количество слоев трансформера

`--tf_nheads` - количество attention голов

`--dim_feedforward` - размерность feedforward модели трансформера

`--tf_maxlen` - максимальная длина последовательности, обрабатываемой трансформером. (только для `agent.py`)

### Пример запуска: </br>
`python agent.py --use_lstm --num_actors 16 --batch_size 8
--unroll_length 80 --learning_rate 0.0001
--entropy_cost 0.001  --epsilon 0.001
--savedir ./test/ --tf_layers 4 --tf_nheads 4
--dim_feedforward 32 --tf_maxlen 180`
