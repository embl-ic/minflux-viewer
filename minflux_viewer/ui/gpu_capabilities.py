"""Startup OpenGL capability and memory-budget probing.

Qt 6 has no automatic ANGLE fallback on Windows, so importing PyOpenGL is not
enough to decide whether an OpenGL attribute renderer can be offered.  The app
creates a small off-screen context once, immediately after ``QApplication``,
and stores the result in :class:`AppState` for menus and viewers to consult.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil

# GLScatterPlotItem uploads one vec3 position and one vec4 colour per marker.
GPU_BYTES_PER_POINT = (3 + 4) * 4
# CPU construction retains the two float64 source coordinates and creates the
# float32 position/colour upload arrays. This excludes canonical dataset arrays,
# which already exist independently of the renderer.
CPU_WORK_BYTES_PER_POINT = 2 * 8 + GPU_BYTES_PER_POINT


@dataclass(frozen=True)
class GpuCapabilities:
    available: bool
    reason: str = ""
    renderer: str = ""
    vendor: str = ""
    free_gpu_memory_bytes: int | None = None
    available_system_memory_bytes: int = 0
    point_limit: int = 0

    @property
    def memory_summary(self) -> str:
        source = self.free_gpu_memory_bytes
        if source:
            return f"{source / (1024 ** 3):.2f} GiB free GPU memory"
        if self.available_system_memory_bytes:
            return (
                f"GPU memory unavailable; using "
                f"{self.available_system_memory_bytes / (1024 ** 3):.2f} GiB "
                "available system memory conservatively"
            )
        return "memory availability unknown"


def point_limit_from_memory(
    *,
    available_system_memory_bytes: int,
    free_gpu_memory_bytes: int | None = None,
) -> int:
    """Derive a safe GL upload limit from current memory, never a point cap.

    Half of reported free VRAM may be used for the persistent VBOs.  CPU-side
    construction is separately limited to one eighth of currently available
    RAM.  On shared-memory/unknown GPUs, the stricter CPU construction budget
    is also used for the GL side.
    """

    system_available = max(0, int(available_system_memory_bytes))
    cpu_budget = system_available // 8
    cpu_limit = cpu_budget // max(1, CPU_WORK_BYTES_PER_POINT)
    if free_gpu_memory_bytes is not None and free_gpu_memory_bytes > 0:
        gpu_budget = int(free_gpu_memory_bytes) // 2
        gpu_limit = gpu_budget // GPU_BYTES_PER_POINT
    else:
        gpu_limit = cpu_budget // GPU_BYTES_PER_POINT
    return max(0, int(min(cpu_limit, gpu_limit)))


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _free_gpu_memory_bytes(gl) -> int | None:
    """Best-effort standard extension queries; shared GPUs may report none."""

    try:
        extensions = _text(gl.glGetString(gl.GL_EXTENSIONS))
    except Exception:
        extensions = ""
    try:
        if "GL_NVX_gpu_memory_info" in extensions:
            # GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, returned in KiB.
            free_kib = int(gl.glGetIntegerv(0x9049))
            if free_kib > 0:
                return free_kib * 1024
        if "GL_ATI_meminfo" in extensions:
            # VBO_FREE_MEMORY_ATI; first integer is free memory in KiB.
            values = gl.glGetIntegerv(0x87FB)
            free_kib = int(values[0] if hasattr(values, "__len__") else values)
            if free_kib > 0:
                return free_kib * 1024
    except Exception:
        return None
    return None


def unavailable_capabilities(reason: str) -> GpuCapabilities:
    system_available = int(psutil.virtual_memory().available)
    return GpuCapabilities(
        available=False,
        reason=str(reason),
        available_system_memory_bytes=system_available,
    )


def probe_gpu_capabilities() -> GpuCapabilities:
    """Create one off-screen OpenGL context and estimate an upload limit."""

    system_available = int(psutil.virtual_memory().available)
    try:
        import pyqtgraph.opengl  # noqa: F401 - verify the shipped backend
        from OpenGL import GL
        from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
    except Exception as exc:  # noqa: BLE001 - capability result, never startup crash
        return unavailable_capabilities(f"OpenGL modules are unavailable: {exc}")

    surface = None
    context = None
    try:
        surface = QOffscreenSurface()
        surface.setFormat(QSurfaceFormat.defaultFormat())
        surface.create()
        if not surface.isValid():
            return unavailable_capabilities("Qt could not create an off-screen surface")
        context = QOpenGLContext()
        context.setFormat(surface.format())
        if not context.create() or not context.isValid():
            return unavailable_capabilities("Qt could not create an OpenGL context")
        if not context.makeCurrent(surface):
            return unavailable_capabilities("Qt could not make the OpenGL context current")
        renderer = _text(GL.glGetString(GL.GL_RENDERER))
        vendor = _text(GL.glGetString(GL.GL_VENDOR))
        version = _text(GL.glGetString(GL.GL_VERSION))
        if not renderer or not version:
            return unavailable_capabilities("the OpenGL context returned no renderer/version")
        free_gpu = _free_gpu_memory_bytes(GL)
        limit = point_limit_from_memory(
            available_system_memory_bytes=system_available,
            free_gpu_memory_bytes=free_gpu,
        )
        if limit <= 0:
            return unavailable_capabilities("insufficient memory for an OpenGL point buffer")
        return GpuCapabilities(
            available=True,
            renderer=renderer,
            vendor=vendor,
            free_gpu_memory_bytes=free_gpu,
            available_system_memory_bytes=system_available,
            point_limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 - report capability, keep CPU app usable
        return unavailable_capabilities(f"OpenGL probe failed: {exc}")
    finally:
        try:
            if context is not None:
                context.doneCurrent()
        except Exception:
            pass
        # Keep explicit references until after doneCurrent(), then destroy the
        # context before the surface it was current against. Reversing that
        # lifetime can trip a Windows OpenGL-driver heap error at process exit.
        context = None
        surface = None
