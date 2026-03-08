# LLM of kaggle competition

本次目标是，根据Chatbot Arena的两个模型比较来评分，比如有两个模型model a，model b，为了比较两个模型的输出回答优劣，会有许多用户进行提问问题，之后根据两个模型输出回答选择更好（因为只要不是错误的乱回答，与其说是更好，不如说是更偏好哪种回答），model a更好？model b更好？model a与model b平局？

## NLP Encoder（DeBERTa）

### Problem:

1.每次截断文本都是末尾，但是一般末尾才是总结，才是真的核心问题，那岂不是有很大概率把最重要的论点论据给截断了。

