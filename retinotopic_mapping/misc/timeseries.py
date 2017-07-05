import json
from jsonschema import validate
import traceback

class TimeseriesValidator(object):
    schema = \
    {
      "$schema": "http://json-schema.org/draft-04/schema",
      "version":  0.1,
      "definitions": {
        "header": {
          "type": "object",
          "description": "Header defining how to name and lookup data",
          "anyOf": [
            { "required": [ "columns" ] },
            { "required": [ "event_data" ] }
          ],
          "properties": {
            "columns": { 
              "type": "object",
              "patternProperties": {
                "^[0-9]+$": { "$ref": "#/definitions/column" }
              },
              "additionalProperties": False
            },
            "event_data": {
              "type": "array",
              "description": "Event-based data references. Each item should reference a time_ref which has event times.",
              "items": { "type": "string" }
            }
          }
        },
        "column": {
          "type": "object",
          "required": [ "name" ],
          "properties": {
            "name": { "type": "string" },
            "time_ref": { "type": "string" }
          }
        },
        "time_ref": {
          "type": "object",
          "description": "Time axis information, specified either with timestamps or sampling rate",
          "oneOf": [
            { "required": [ "timestamps", "time_units" ] },
            { "required": [ "sampling_rate", "rate_units" ] }
          ],
          "properties": {
            "timestamps": {
              "type": "array",
              "items": {
                "type": [ "integer", "number", "string" ]
              }
            },
            "sampling_rate": { "type": "number" },
            "time_units": { "type": "string" },
            "rate_units": { "type": "string"},
            "start_time": { "type": "string" },
            "end_time": { "type":  "string"}
          }
        }
      },
      "type": "object",
      "anyOf": [
        { "required": [ "header", "data" ] },
        { "required": [ "header", "time_info" ] },
      ],
      "properties": {
        "header": { "$ref": "#/definitions/header" },
        "data": {
          "type": "array",
          "description": "A data array, currently any type",
          "items": { }
        },
        "time_info": {
          "type": "object",
          "description": "Dictionary of time_ref objects for looking up time axis information",
          "additionalProperties": { "$ref": "#/definitions/time_ref" }
        }
      }
    }

    @staticmethod
    def is_valid(obj):
        """Check if `obj` adheres to the Timeseries schema"""
        try:
            validate(obj, TimeseriesValidator.schema)
            return True
        except Exception as e:
            print(traceback.format_exc(e))
            return False


test_data1 = {
    "header": {
        "columns": {
            "0": {"name": "x",
                  "time_ref": "t1"},
            "1": {"name": "y"}
        },
        "event_data": ["t2"]
    },
    "data": [[0, 1, 2, 3, 4, 5, 6], [-1.5, 4.6, 8.5, 9.9, 10.0, 5.3, 100]],
    "time_info": {
        "t1": {
            "sampling_rate": 0.5,
            "rate_units": "Hz",
        },
        "t2": {
            "timestamps": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "time_units": "s"
        }
    }
}

if __name__ == "__main__":
    print("Validating test data")
    print TimeseriesValidator.is_valid(test_data1)