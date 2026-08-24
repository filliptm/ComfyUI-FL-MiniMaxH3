function startsCanvasGesture(event, canvas) {
  return event.button === 1 || (
    event.button === 0
    && (
      canvas.read_only
      || (canvas.dragZoomEnabled && event.ctrlKey && event.shiftKey && !event.altKey)
    )
  );
}

export function addCanvasNavigation(element, canvas) {
  const graphCanvas = canvas.canvas;
  let pointerId = null;

  element.addEventListener("wheel", (event) => {
    graphCanvas.dispatchEvent(new WheelEvent(event.type, event));
    event.preventDefault();
    event.stopPropagation();
  }, { passive: false });

  element.addEventListener("pointerdown", (event) => {
    if (!startsCanvasGesture(event, canvas)) return;
    pointerId = event.pointerId;
    graphCanvas.dispatchEvent(new PointerEvent(event.type, event));
    event.preventDefault();
    event.stopPropagation();
  });

  for (const eventName of ["pointermove", "pointerup", "pointercancel"]) {
    element.addEventListener(eventName, (event) => {
      if (event.pointerId !== pointerId) return;
      graphCanvas.dispatchEvent(new PointerEvent(event.type, event));
      if (eventName !== "pointermove") pointerId = null;
      event.preventDefault();
      event.stopPropagation();
    });
  }

  element.addEventListener("auxclick", (event) => {
    if (event.button === 1) event.preventDefault();
  });
}
