def create_function_info_from_pydantic(pydantic_object, description: str = None):
    """
    Generates a structured JSON schema for a pydantic model.

    Parameters
    ----------
    pydantic_object : BaseModel
        The Pydantic model to generate a schema for.
    description : str
        A description of the Pydantic object.
    """
    if description is None:
        description = "No description provided."

    schema = pydantic_object.schema()
    properties = {}
    required_fields = []
    for name, details in schema["properties"].items():
        properties[name] = {
            "type": details["type"],
            "description": details.get("description", "No description provided."),
        }
        required_fields.append(name)

    if hasattr(pydantic_object.Config, "schema_extra"):
        schema_extra = pydantic_object.Config.schema_extra
        if "properties" in schema_extra:
            for name, details in schema_extra["properties"].items():
                properties[name].update(details)

    info = {
        "type": "function",
        "function": {
            "name": pydantic_object.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
            },
        },
    }
    return info
