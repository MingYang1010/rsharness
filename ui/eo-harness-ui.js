(() => {
  const MAX_ATTEMPTS = 100;

  function findTerria() {
    const root = document.querySelector("#ui");
    const rootKey = Object.keys(root || {}).find((key) =>
      key.startsWith("__reactContainer$")
    );
    if (!rootKey) return undefined;

    const stack = [root[rootKey]];
    const seen = new Set();
    while (stack.length > 0) {
      const fiber = stack.pop();
      if (!fiber || typeof fiber !== "object" || seen.has(fiber)) continue;
      seen.add(fiber);

      for (const props of [fiber.memoizedProps, fiber.pendingProps]) {
        const terria = props?.terria || props?.viewState?.terria;
        if (terria?.currentViewer && terria?.mapNavigationModel) return terria;
      }

      if (fiber.child) stack.push(fiber.child);
      if (fiber.sibling) stack.push(fiber.sibling);
    }
    return undefined;
  }

  function exposeZoomOnSmallScreens() {
    const terria = findTerria();
    const zoomItem = terria?.mapNavigationModel?.items?.find(
      (item) => item.id === "zoom"
    );
    if (!zoomItem) return false;

    // TerriaJS 8.12.2 registers ZoomControl as desktop-only.
    zoomItem.screenSize = "any";
    return true;
  }

  function labelLanguageButton() {
    const button = document.querySelector(".tjs-menu-bar__langBtn");
    if (!button) return;

    const label = "切换语言 / Change language";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
  }

  function installZoomRenderBridge() {
    const marker = "eoHarnessZoomRenderBridge";
    if (document.documentElement.dataset[marker]) return;

    document.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target : undefined;
      if (!target?.closest('[class*="ZoomControl__StyledZoomControl"]')) return;

      const terria = findTerria();
      if (!terria) return;

      const renderUntil = performance.now() + 350;
      const requestFrame = () => {
        terria.currentViewer?.notifyRepaintRequired?.();
        terria.currentViewer?.scene?.requestRender?.();
        if (performance.now() < renderUntil) {
          window.requestAnimationFrame(requestFrame);
        }
      };
      window.requestAnimationFrame(requestFrame);
    });

    document.documentElement.dataset[marker] = "true";
  }

  installZoomRenderBridge();

  let attempts = 0;
  const timer = window.setInterval(() => {
    attempts += 1;
    labelLanguageButton();
    if (exposeZoomOnSmallScreens() || attempts >= MAX_ATTEMPTS) {
      window.clearInterval(timer);
    }
  }, 200);
})();
