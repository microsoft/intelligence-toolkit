# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from javascript.styles import add_styles
import components.app_user as au
import components.app_terminator as at
import components.app_openai as ao

def load_multipage_app():
    #Load user if logged in
    user = au.app_user()
    user.view_get_info()

    #Terminate app (if needed for .exe)
    terminator = at.app_terminator()
    terminator.terminate_app_btn()

    #OpenAI key set
    app_openai = ao.app_openai()
    app_openai.api_info()

    #load css
    # add_styles()

