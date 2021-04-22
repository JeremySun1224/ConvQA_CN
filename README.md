# ConvQA_CN
基于抽取式模型和生成式模型解决多轮对话式机器阅读理解任务。

数据集样例如下：

```python
{
            "Source": "儿童故事",
            "Passage": "圣诞前夕(12月24日)也就是常说的「平安夜」，在这天晚上总会看到一群可爱的小男生或小女生，手拿诗歌弹着吉他，走在冰冰的雪地上，一家一家的唱着诗歌报佳音。到底佳音队这种节日活动是怎么来的呢？\n\n话说耶稣诞生的那一晚，在旷野看守羊群的牧羊人突然听见天上有声音发出，向他们报耶稣降生的消息，因为据圣经记载，耶稣来是要作世人心中的王。因此天使便通过这些牧羊人把消息传给更多的人知道。\n\n后来人们为了把耶稣降生的消息传给大家知道，就效仿天使，在平安夜的晚上到处去向人传讲耶稣降生的消息。直到今日，报佳音已经变成圣诞不可缺少的一个节目。通常佳音队是由大约二十名青年人，加上一个装扮成天使的小女孩和一位圣诞老人组成。然后在平安夜晚大约是九点过后开始一家一家的去报佳音。\n\n每当佳音队去到一个家庭时，先会唱几首大家都熟悉的圣诞歌曲，然后再由小女孩念出圣经的话语让该家庭知道今夜是耶稣降生的日子，过后大家一起祷告再唱一两首诗歌，再由慷慨大方的圣诞老人派送圣诞礼物给该家庭的小孩，整个报佳音的过程就完成了！\n\n整个报佳音的过程大约要到第二天凌晨四点左右才结束。当然这也要看各国的庆祝方式。",
            "Questions": [
                {
                    "input_text": "圣诞前夕是什么时候？",
                    "turn_id": 1,
                    "question_type": "factoid"
                },
                {
                    "input_text": "圣诞前夕通常叫什么？",
                    "turn_id": 2,
                    "question_type": "factoid"
                },
                {
                    "input_text": "在这天晚上通常会看见什么？",
                    "turn_id": 3,
                    "question_type": "factoid"
                },
                {
                    "input_text": "在什么时候牧羊人听见天上有声音发出？",
                    "turn_id": 4,
                    "question_type": "factoid"
                },
                {
                    "input_text": "据圣经记载，耶稣是来干什么的？",
                    "turn_id": 5,
                    "question_type": "factoid"
                },
                {
                    "input_text": "什么是圣诞节必不可缺少的节目？",
                    "turn_id": 6,
                    "question_type": "factoid"
                },
                {
                    "input_text": "通常佳音队是怎么组成的？",
                    "turn_id": 7,
                    "question_type": "why"
                },
                {
                    "input_text": "什么时候佳音队开始一家一家的报佳音？",
                    "turn_id": 8,
                    "question_type": "factoid"
                },
                {
                    "input_text": "整个报佳音的过程大约到什么时候结束？",
                    "turn_id": 9,
                    "question_type": "factoid"
                }
            ],
            "Answers": [
                {
                    "input_text": "12月24号。",
                    "turn_id": 1,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                },
                {
                    "input_text": "平安夜。",
                    "turn_id": 2,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                },
                {
                    "input_text": "这天晚上总会看见一群男生或女生，在雪地上唱诗歌报佳音。",
                    "turn_id": 3,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                },
                {
                    "input_text": "在耶稣诞生的前一晚。",
                    "turn_id": 4,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                },
                {
                    "input_text": "耶稣是来做是人心中的王。",
                    "turn_id": 5,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                },
                {
                    "input_text": "报佳音。",
                    "turn_id": 6,
                    "answer_id": 1,
                    "answer_type": "extractive",
                    "answer_start": [
                        73,
                        324
                    ]
                },
                {
                    "input_text": "佳音队是由大约二十名青年人，加上一个盛装打扮的天使和一个圣诞老人。",
                    "turn_id": 7,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                },
                {
                    "input_text": "在平安夜晚九点过后。",
                    "turn_id": 8,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                },
                {
                    "input_text": "整个过程大约到第二天凌晨四年左右才结束。",
                    "turn_id": 9,
                    "answer_id": 1,
                    "answer_type": "generative",
                    "answer_start": [
                        -3
                    ]
                }
            ],
            "answer_statistics": [
                {
                    "answer_count": 9,
                    "span_count": 1,
                    "nospan_count": 8,
                    "yesno_count": 0,
                    "cannotanswer_count": 0,
                    "span_ratio": 11.11111111111111,
                    "nospan_ratio": 88.88888888888889,
                    "yesno_ratio": 0.0,
                    "cannotanswer_ratio": 0.0
                }
            ],
            "question_statistics": [
                {
                    "question_count": 9,
                    "coreference_count": 1,
                    "omitting_count": 0,
                    "factual_count": 8,
                    "yesno_count": 0,
                    "why_count": 1,
                    "coreference_ratio": 11.11111111111111,
                    "omitting_ratio": 0.0,
                    "factual_ratio": 88.88888888888889,
                    "yesno_ratio": 0.0,
                    "why_ratio": 11.11111111111111
                }
            ],
            "id": 230
        }
```

生成式模型的预测结果：

```python
{
        "id": "230", 
        "turn_id": 1, 
        "answer": "12月24日"
    }, 
    {
        "id": "230", 
        "turn_id": 2, 
        "answer": "平安夜 。"
    }, 
    {
        "id": "230", 
        "turn_id": 3, 
        "answer": "一群可爱的小男生或小女生 。"
    }, 
    {
        "id": "230", 
        "turn_id": 4, 
        "answer": "耶稣诞生的那一晚 。"
    }, 
    {
        "id": "230", 
        "turn_id": 5, 
        "answer": "他 是要作世人心中的王。"
    }, 
    {
        "id": "230", 
        "turn_id": 6, 
        "answer": "报佳音。"
    }, 
    {
        "id": "230", 
        "turn_id": 7, 
        "answer": "大约二十名青年人，加上一个装扮成天使的小女孩和一位圣诞老人组成。"
    }, 
    {
        "id": "230", 
        "turn_id": 8, 
        "answer": "平安夜晚大约是九点过后 。"
    }, 
    {
        "id": "230", 
        "turn_id": 9, 
        "answer": "一个家庭 。"
    }
```

抽取式模型的预测结果：

```python
    {
        "id": "230",
        "turn_id": 1,
        "answer": "12月 24日"
    },
    {
        "id": "230",
        "turn_id": 2,
        "answer": "无法作答"
    },
    {
        "id": "230",
        "turn_id": 3,
        "answer": "无法作答"
    },
    {
        "id": "230",
        "turn_id": 4,
        "answer": "无法作答"
    },
    {
        "id": "230",
        "turn_id": 5,
        "answer": "无法作答"
    },
    {
        "id": "230",
        "turn_id": 6,
        "answer": "是的"
    },
    {
        "id": "230",
        "turn_id": 7,
        "answer": "无法作答"
    },
    {
        "id": "230",
        "turn_id": 8,
        "answer": "无法作答"
    },
    {
        "id": "230",
        "turn_id": 9,
        "answer": "无法作答"
    }
```

数据集无法提供，对模型方法感兴趣的可以移步[CoQA](https://stanfordnlp.github.io/coqa/)展开相关实验。