"""The render tile workers run on their own named pool.

Every render window used to set the maximum thread count of Qt's GLOBAL pool
to four, so anything else submitted there was capped by a render view having
been opened, and every render window shared those same four threads.
"""

from __future__ import annotations


def test_a_render_scheduler_leaves_the_global_pool_alone(qapp):
    from PyQt6.QtCore import QThreadPool

    from minflux_viewer.ui.background_tasks import shared_thread_pool
    from minflux_viewer.ui.render_scheduler import RenderScheduler

    global_pool = QThreadPool.globalInstance()
    before = global_pool.maxThreadCount()
    global_pool.setMaxThreadCount(13)          # distinctive, so a reset shows
    try:
        scheduler = RenderScheduler()
        assert global_pool.maxThreadCount() == 13
        assert scheduler._pool is not global_pool
        assert scheduler._pool is shared_thread_pool(RenderScheduler.POOL)
    finally:
        global_pool.setMaxThreadCount(before)
