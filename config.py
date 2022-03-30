
from dynaconf import Dynaconf

import os


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[os.environ['SETTINGS_FILE'], '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
