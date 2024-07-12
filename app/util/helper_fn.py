# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import os

from util.enums import Mode

mode = os.getenv("MODE", Mode.DEV.value)


def appInDevMode():
    return mode == Mode.DEV.value


def appInExeMode():
    return mode == Mode.EXE.value
