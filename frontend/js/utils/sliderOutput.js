export function bindSliderOutput(slider, outputEl, { format = (v) => v, onChange } = {}) {
  function sync() {
    outputEl.textContent = format(slider.value);
    onChange?.(Number(slider.value));
  }

  slider.addEventListener("input", sync);
  sync();
}
