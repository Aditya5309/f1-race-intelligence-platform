"""AppTest entry script for tests/test_race_center_appTest.py.

AppTest.from_function() re-executes only a function's own source lines,
with no access to its enclosing module's imports (page_header,
sidebar_model_panel, etc.) -- so this is a real, minimal script file
instead, run via AppTest.from_file(), matching how scripts/smoke.py
already drives the full dashboard.py the same way.
"""

from app.views import race_center

race_center.render()
