"""pytest 公共夹具。"""

import pytest


@pytest.fixture(autouse=True)
def isolated_module_registry():
    """每个测试后还原进程级模块目录（运行期注册是内核状态，测试之间不能泄漏）。"""
    from avm import modules

    catalog = dict(modules.MODULE_CATALOG)
    descriptors = dict(modules.REGISTERED_MODULE_DESCRIPTORS)
    views = dict(modules._SHARED_VIEWS)
    modules._invalidate_catalog_caches()
    try:
        yield
    finally:
        modules.MODULE_CATALOG.clear()
        modules.MODULE_CATALOG.update(catalog)
        modules.REGISTERED_MODULE_DESCRIPTORS.clear()
        modules.REGISTERED_MODULE_DESCRIPTORS.update(descriptors)
        modules._SHARED_VIEWS.clear()
        modules._SHARED_VIEWS.update(views)
        modules._invalidate_catalog_caches()
