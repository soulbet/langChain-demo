|异常类型|处理方案|核心 API|补充说明|
|-|-|-|-|
|**工具执行报错**|错误信息返回给 LLM 自行修正|`handle_tool_error=True` / 自定义函数|让 LLM 根据错误信息调整参数或换工具，是实现“自主纠错”的关键|
|**模型调用失败**|自动切换备用模型|`with_fallbacks([备用模型])`|适合处理模型服务短暂不可用。支持 Chat/Embedding 等所有 Runnable|
|**网络/限流错误**|指数退避重试|`RetryPolicy(max_attempts, retry_on)`|重试策略的核心，需配置重试条件和退避间隔|
|**自定义重试条件**|按 HTTP 状态码等动态判断|`RetryPolicy(retry_on=函数)`|可根据异常类型或返回值自定义是否需要重试|
|**LCEL 链异常**|局部容错 + 链级降级|`RunnableLambda(try-except)` + `with_fallbacks`|细粒度控制链中每个环节的失败处理|
|**图级别错误**|错误计数 + 优雅退出|状态字段追踪 + 条件路由|在 LangGraph 中通过节点状态计数，达到阈值后路由到“降级节点”|
|**超时**|超时 + fallback|`config` 中设置 / `try-except` 包裹|需区分步骤超时和整体超时，防止 Agent 无限等待|
