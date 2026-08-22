export function wireDisclosureControls(moduleRoot) {
  const detailsEls = moduleRoot.querySelectorAll("details.tech");
  const expandBtn = moduleRoot.querySelector(".expand-all");
  const collapseBtn = moduleRoot.querySelector(".collapse-all");

  expandBtn?.addEventListener("click", () => {
    detailsEls.forEach((details) => { details.open = true; });
  });

  collapseBtn?.addEventListener("click", () => {
    detailsEls.forEach((details) => { details.open = false; });
  });
}
