Собирает MoE модель из базовых моделей-доноров. Генерирует modular и modeling-файлы, загружает веса в новую модель и сохраняет ее (config + tokenizer + model.safetensors).<br><br>**Чтобы использовать надо:**<br>
Установить версию transformers с гитхаба. (`pip install transformers` не установит необходимый `utils/modular_model_converter.py`)
```
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
Установить зависимости:
```
pip install torch
pip install libcst
pip install ruff
```

**Использование:**
```
python moe_generator.py <spec.yaml> [--move_to_cwd]
```
```
    <spec.yaml> - YAML файл со спецификацией желаемой модели
    [--move_to_cwd] - опциональный. Если указан - после генерации переместит
                      директорию с моделинг файлами модели в текущую директорию,
                      преобразовав относительные импорты в абсолютные.
```
Пример <spec.yaml>:
```
name: MyQwen2Moe
attention: Qwen2.5-0.5B-Instruct
experts:
  - Qwen2.5-0.5B-Instruct
  - Qwen2.5-0.5B-Instruct
  - Qwen2.5-0.5B-Instruct
num_experts_per_tok: 2
moe_layers_idx: [21, 22, 23]
```
```
name - имя-префикс модели. На его основе формируются имена директорий и классов.
attention - путь до директории с model.safetensors модели.
experts - список директорий с весами моделей - доноров mlp для экспертов
num_experts_per_tok - количество активируемых экспертов на токен
moe_layers_idx - индексы слоев с MoE вместо обычного MLP```
