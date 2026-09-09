# PCB知识库

把需要问答系统检索的资料放在本目录中，然后运行：

```bash
python build_rag_index.py
```

支持 `.md`、`.txt`、`.pdf`、`.docx` 和 `.pptx`。项目根目录的 `README.md` 也会默认进入知识库。

知识文档应只包含可信的项目说明、AOI操作规范、缺陷判定标准和培训资料。回答会显示文档名称以及PDF/PPT页码。
