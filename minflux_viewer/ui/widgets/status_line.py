"""A one-line status read-out that never widens its window.

A ``QLabel`` asks for its whole text as its minimum width. A view's status
line can be long -- in Ortho-View it carries the 3-D coordinate, about 890 px
at the status font -- so writing it made the window grow past the size the
user, or the Ortho-View screen fit, had given it, and every pane's view range
moved with it. :class:`StatusLine` asks for no width of its own: it shows as
much of the text as fits, ending in an ellipsis when it is cut, and carries
the full text as its tooltip. ``text()`` still returns the full text, so code
that appends to the status line keeps working. Its height is one line of its
font whether or not it holds text, so the panes above it do not move when the
first status arrives.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QLabel, QSizePolicy, QStyle, QStyleOption


class StatusLine(QLabel):
    """A single-line ``QLabel`` whose width is set by its window, not its text."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setText(text)

    def setText(self, text: str | None) -> None:  # noqa: N802 (Qt override)
        text = text or ""
        super().setText(text)
        self.setToolTip(text)

    def _line_height(self) -> int:
        # One line of the current font, whatever the text: an empty QLabel
        # asks for a different height than a filled one, so the layout above
        # would move by a pixel or two the moment the first status arrived.
        margins = self.contentsMargins()
        return (self.fontMetrics().height() + margins.top() + margins.bottom()
                + 2 * self.margin())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(super().sizeHint().width(), self._line_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(0, self._line_height())

    def displayed_text(self) -> str:
        """What fits in the current width: the text, or its start and '…'."""
        return self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideRight, max(0, self.contentsRect().width()))

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        style = self.style()
        if style is None:
            return
        painter = QPainter(self)
        option = QStyleOption()
        option.initFrom(self)
        style.drawPrimitive(QStyle.PrimitiveElement.PE_Widget, option, painter, self)
        style.drawItemText(
            painter, self.contentsRect(), int(self.alignment()), self.palette(),
            self.isEnabled(), self.displayed_text(), self.foregroundRole())
