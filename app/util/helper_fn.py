# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

from util.enums import Mode

mode = os.getenv("MODE", Mode.DEV.value)


def app_in_dev_mode():
    return mode == Mode.DEV.value


def app_in_exe_mode():
    return mode == Mode.EXE.value
