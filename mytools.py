import fnmatch
import os
from typing import Optional, Type
import re

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)


class FileSearchInput(BaseModel):
    """Input for FileSearchTool."""

    dir_path: str = Field(
        default=".",
        description="Subdirectory to search in.",
    )
    pattern: str = Field(
        ...,
        description="Unix shell regex, where * matches everything.",
    )

class FileSearchTool(BaseFileToolMixin, BaseTool):
    """Tool that searches for files in a subdirectory that contain matches of a given regex pattern."""

    name: str = "file_search"
    args_schema: Type[BaseModel] = FileSearchInput # type: ignore
    description: str = (
        "Recursively search for files in a subdirectory that contain matches of the regex pattern"
    )

    def _run(
        self,
        pattern: str,
        dir_path: str = ".",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            dir_path_ = self.get_relative_path(dir_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="dir_path", value=dir_path)
        matches = []
        try:
            for root, _, filenames in os.walk(dir_path_):
                for filename in filenames:
                    absolute_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(absolute_path, dir_path_)
                    with open(absolute_path,encoding="latin-1") as f:
                        if len(re.findall(pattern,f.read()))>0:
                            #print(f"\n{absolute_path}\n{relative_path}")
                            matches.append(absolute_path)
                for filename in fnmatch.filter(filenames, pattern):
                    absolute_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(absolute_path, dir_path_)
                    matches.append(absolute_path)
            if matches:
                return "\n".join(matches)
            else:
                return f"No files found for pattern {pattern} in directory {dir_path}"
        except Exception as e:
            return "Error: " + str(e)

    # TODO: Add aiofiles method