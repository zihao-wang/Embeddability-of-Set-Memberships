
def get_embedding_module(module_name):
    if module_name == "tabular":
        from .tabular import TabularEmbedding
        return TabularEmbedding
    else:
        # also a default module name
        from .tabular import TabularEmbedding
        return TabularEmbedding