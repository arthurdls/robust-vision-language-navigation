"""Shared pytest configuration for rvln tests.

Unregisters the ROS launch_testing_ros plugin, which defines a non-standard
hook (pytest_launch_collect_makemodule) that causes a PluginValidationError
with newer versions of pluggy/pytest.
"""


def pytest_configure(config):
    pm = config.pluginmanager
    for name in list(pm.list_name_plugin()):
        plugin_name, plugin = name
        mod = getattr(plugin, "__name__", "") or ""
        if "launch_testing" in mod or "launch_testing" in (plugin_name or ""):
            if plugin is not None:
                pm.unregister(plugin)
            elif plugin_name is not None:
                pm.unregister(name=plugin_name)
