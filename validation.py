import jsonschema

def matches_schema(schema: dict, data: dict) -> bool:
    """
    Validates campaign recommendations data against the schema.
    Returns True if valid, raises ValidationError if invalid.
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        return False