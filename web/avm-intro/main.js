(() => {
  'use strict';

  const scenes = Array.from(document.querySelectorAll('.scene'));
  const sceneCount = scenes.length;
  const chapterNav = Array.from(document.querySelectorAll('.chapter-nav__item'));
  const main = document.getElementById('main');
  const liveRegion = document.getElementById('liveRegion');
  const currentSection = document.getElementById('currentSection');
  const currentIndex = document.getElementById('currentIndex');
  const pagerFill = document.getElementById('pagerFill');
  const pagerTicks = document.getElementById('pagerTicks');
  const prevButton = document.getElementById('prevButton');
  const nextButton = document.getElementById('nextButton');
  const fullscreenButton = document.getElementById('fullscreenButton');
  const menuButton = document.getElementById('menuButton');
  const scrollCue = document.getElementById('scrollCue');
  const menuDialog = document.getElementById('menuDialog');
  const menuPanel = menuDialog.querySelector('.menu__panel');
  const menuCloseButton = menuDialog.querySelector('button[data-close-menu]');
  const tabs = Array.from(document.querySelectorAll('.tab'));
  const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
  const scenarios = Array.from(document.querySelectorAll('.scenario'));
  const baseTags = Array.from(document.querySelectorAll('.base-bar__tags [data-base]'));

  let current = 0;
  let leavingTimer = 0;
  let lastWheelAt = 0;
  let wheelAccumulator = 0;
  let wheelResetTimer = 0;
  let touchStartX = 0;
  let touchStartY = 0;
  let touchStartTarget = null;
  let menuOpen = false;
  let returnFocus = null;
  let tabFocus = 0;
  let previousTabFocus = 0;
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const pad = (number) => String(number).padStart(2, '0');

  function readHashIndex() {
    const raw = window.location.hash.replace(/^#/, '');
    if (!/^\d+$/.test(raw)) return null;
    const index = Number(raw);
    return Number.isInteger(index) && index >= 0 && index < sceneCount ? index : null;
  }

  function setChromeForScene(scene) {
    document.body.dataset.chrome = scene.classList.contains('scene--dark') ? 'dark' : 'light';
  }

  function syncInert() {
    scenes.forEach((scene, index) => {
      const isCurrent = index === current;
      scene.toggleAttribute('inert', !isCurrent);
    });
  }

  function updateTicks() {
    Array.from(pagerTicks.children).forEach((tick, index) => {
      tick.classList.toggle('is-past', index < current);
      tick.classList.toggle('is-current', index === current);
      tick.setAttribute('aria-current', index === current ? 'step' : 'false');
    });
  }

  function updateChrome() {
    const scene = scenes[current];
    const title = scene.dataset.title || '';
    const chapterIndex = Number(scene.dataset.scene);

    currentSection.textContent = title;
    currentIndex.textContent = `${pad(current)} / ${pad(sceneCount - 1)}`;
    pagerFill.style.width = `${(current / (sceneCount - 1)) * 100}%`;
    prevButton.disabled = current === 0;
    nextButton.disabled = current === sceneCount - 1;

    chapterNav.forEach((item) => {
      const target = Number(item.dataset.go);
      if (target === chapterIndex && target > 0 && target < sceneCount - 1) {
        item.setAttribute('aria-current', 'page');
      } else {
        item.removeAttribute('aria-current');
      }
    });

    updateTicks();
    setChromeForScene(scene);
    liveRegion.textContent = `${title}，第 ${current + 1} 页，共 ${sceneCount} 页。`;
  }

  function clearLeavingScenes() {
    scenes.forEach((scene) => scene.classList.remove('is-leaving'));
  }

  function updateScrollCue() {
    if (menuOpen) {
      document.body.classList.remove('has-scroll-cue');
      return;
    }
    const inner = scenes[current].querySelector('.scene__inner');
    if (!inner) return;
    const canScroll = inner.scrollHeight > inner.clientHeight + 4;
    const atBottom = inner.scrollTop + inner.clientHeight >= inner.scrollHeight - 4;
    document.body.classList.toggle('has-scroll-cue', canScroll && !atBottom);
  }

  function wireScrollCue() {
    scenes.forEach((scene) => {
      const inner = scene.querySelector('.scene__inner');
      if (!inner) return;
      inner.addEventListener('scroll', () => {
        if (scenes[current] === scene) updateScrollCue();
      }, { passive: true });
    });
  }

  function goTo(index, options = {}) {
    const { push = true, focusMain = false } = options;
    const target = Math.max(0, Math.min(index, sceneCount - 1));
    if (target === current && !options.force) return;

    const oldScene = scenes[current];
    const newScene = scenes[target];
    const scrollTarget = newScene.querySelector('.scene__inner');

    window.clearTimeout(leavingTimer);
    clearLeavingScenes();

    if (oldScene !== newScene) {
      oldScene.classList.remove('is-active');
      oldScene.classList.add('is-leaving');
      oldScene.setAttribute('aria-hidden', 'true');
      oldScene.toggleAttribute('inert', true);
    }

    newScene.classList.remove('is-leaving');
    newScene.classList.add('is-active');
    newScene.setAttribute('aria-hidden', 'false');
    newScene.toggleAttribute('inert', false);
    if (scrollTarget) scrollTarget.scrollTop = 0;

    current = target;
    updateChrome();
    syncInert();
    window.requestAnimationFrame(updateScrollCue);

    leavingTimer = window.setTimeout(() => {
      oldScene.classList.remove('is-leaving');
    }, prefersReducedMotion.matches ? 20 : 520);

    if (push) {
      const nextHash = `#${current}`;
      if (window.location.hash !== nextHash) {
        window.history.pushState({ scene: current }, '', nextHash);
      }
    }

    if (focusMain) {
      main.focus({ preventScroll: true });
    }
  }

  function createTicks() {
    const fragment = document.createDocumentFragment();
    scenes.forEach((scene, index) => {
      const tick = document.createElement('button');
      tick.type = 'button';
      tick.setAttribute('aria-label', `前往：${scene.dataset.title || `第 ${index + 1} 页`}`);
      tick.style.left = `${(index / (sceneCount - 1)) * 100}%`;
      tick.addEventListener('click', () => goTo(index));
      fragment.appendChild(tick);
    });
    pagerTicks.appendChild(fragment);
  }

  function initSceneState() {
    const fromHash = readHashIndex();
    const initial = fromHash === null ? 0 : fromHash;

    scenes.forEach((scene, index) => {
      scene.classList.toggle('is-active', index === initial);
      scene.classList.remove('is-leaving');
      scene.setAttribute('aria-hidden', index === initial ? 'false' : 'true');
      scene.toggleAttribute('inert', index !== initial);
      const inner = scene.querySelector('.scene__inner');
      if (inner) inner.scrollTop = 0;
    });

    current = initial;
    updateChrome();
    setChromeForScene(scenes[current]);

    if (fromHash === null) {
      window.history.replaceState({ scene: current }, '', `#${current}`);
    }
  }

  /* -----------------------------------------------------------------------
     Tabs
     ----------------------------------------------------------------------- */

  function setTab(nextIndex, focusTab = false) {
    const safeIndex = Math.max(0, Math.min(nextIndex, tabs.length - 1));
    tabs.forEach((tab, index) => {
      const selected = index === safeIndex;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected) previousTabFocus = index;
    });
    tabPanels.forEach((panel, index) => {
      const selected = index === safeIndex;
      panel.hidden = !selected;
      panel.classList.toggle('is-active', selected);
    });
    tabFocus = safeIndex;
    if (focusTab) tabs[safeIndex].focus();
    window.requestAnimationFrame(updateScrollCue);
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => setTab(index));
    tab.addEventListener('keydown', (event) => {
      let nextIndex = null;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') nextIndex = 0;
      if (event.key === 'End') nextIndex = tabs.length - 1;
      if (nextIndex === null) return;
      event.preventDefault();
      event.stopPropagation();
      setTab(nextIndex, true);
    });
  });

  // Ensure all SVG paths used for the stroke animation have a normalized length.
  document.querySelectorAll('.draw-line').forEach((path) => {
    if (!path.hasAttribute('pathLength')) path.setAttribute('pathLength', '1');
  });

  /* -----------------------------------------------------------------------
     Chapter 5 — base mechanism highlighting
     ----------------------------------------------------------------------- */

  function highlightBase(tokens) {
    const tokenSet = new Set(tokens);
    baseTags.forEach((tag) => {
      tag.classList.toggle('is-lit', tokenSet.has(tag.dataset.base));
    });
  }

  scenarios.forEach((scenario) => {
    const tokens = (scenario.dataset.match || '').split(/\s+/).filter(Boolean);
    scenario.addEventListener('pointerenter', () => highlightBase(tokens));
    scenario.addEventListener('focusin', () => highlightBase(tokens));
    scenario.addEventListener('pointerleave', () => highlightBase([]));
    scenario.addEventListener('focusout', (event) => {
      if (!scenario.contains(event.relatedTarget)) highlightBase([]);
    });
  });

  /* -----------------------------------------------------------------------
     Navigation
     ----------------------------------------------------------------------- */

  document.addEventListener('click', (event) => {
    const go = event.target.closest('[data-go]');
    if (go) {
      event.preventDefault();
      const target = Number(go.dataset.go);
      const cameFromMenu = menuOpen;
      if (cameFromMenu) closeMenu({ restoreFocus: false });
      goTo(target, { focusMain: cameFromMenu });
      return;
    }

    if (event.target.closest('[data-prev]')) {
      goTo(current - 1);
      return;
    }

    if (event.target.closest('[data-next]')) {
      goTo(current + 1);
    }
  });

  prevButton.addEventListener('click', () => goTo(current - 1));
  nextButton.addEventListener('click', () => goTo(current + 1));

  main.addEventListener('touchstart', (event) => {
    if (event.touches.length !== 1) return;
    touchStartX = event.touches[0].clientX;
    touchStartY = event.touches[0].clientY;
    touchStartTarget = event.target;
  }, { passive: true });

  main.addEventListener('touchend', (event) => {
    if (menuOpen || event.changedTouches.length !== 1) return;
    const dx = event.changedTouches[0].clientX - touchStartX;
    const dy = event.changedTouches[0].clientY - touchStartY;
    if (Math.abs(dx) < 64 || Math.abs(dx) < Math.abs(dy) * 1.35) return;

    // A horizontal gesture inside a scrollable diagram should pan the diagram,
    // not switch the chapter.
    const startedOnHorizontalScroller = touchStartTarget instanceof Element
      ? touchStartTarget.closest('.media__canvas, .tab-panel__figure')
      : null;
    if (startedOnHorizontalScroller) {
      const maxScroll = startedOnHorizontalScroller.scrollWidth - startedOnHorizontalScroller.clientWidth;
      const canPan = maxScroll > 3;
      const atLeft = startedOnHorizontalScroller.scrollLeft <= 1;
      const atRight = startedOnHorizontalScroller.scrollLeft >= maxScroll - 1;
      if (canPan && ((dx < 0 && !atRight) || (dx > 0 && !atLeft))) return;
    }

    goTo(current + (dx < 0 ? 1 : -1));
  }, { passive: true });

  function onWheel(event) {
    if (menuOpen) return;
    if (event.ctrlKey || event.metaKey) return;
    if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;

    const activeInner = scenes[current].querySelector('.scene__inner');
    const canScroll = activeInner && activeInner.scrollHeight > activeInner.clientHeight + 3;

    if (canScroll) {
      const atTop = activeInner.scrollTop <= 1;
      const atBottom = activeInner.scrollTop + activeInner.clientHeight >= activeInner.scrollHeight - 2;
      if ((event.deltaY > 0 && !atBottom) || (event.deltaY < 0 && !atTop)) return;
    }

    if (Math.abs(event.deltaY) < 2) return;
    event.preventDefault();

    window.clearTimeout(wheelResetTimer);
    wheelResetTimer = window.setTimeout(() => {
      wheelAccumulator = 0;
    }, 220);

    if (Math.sign(event.deltaY) !== Math.sign(wheelAccumulator)) wheelAccumulator = 0;
    wheelAccumulator += event.deltaY;

    const now = performance.now();
    if (now - lastWheelAt < 850) return;
    if (Math.abs(wheelAccumulator) < 52) return;

    const goingForward = wheelAccumulator > 0;
    wheelAccumulator = 0;
    lastWheelAt = now;
    goTo(current + (goingForward ? 1 : -1));
  }

  window.addEventListener('wheel', onWheel, { passive: false });

  function onKeydown(event) {
    if (event.defaultPrevented) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    const target = event.target;
    if (target instanceof HTMLElement) {
      if (target.isContentEditable) return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
      if (target.closest('[role="tablist"]') && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
    }

    if (menuOpen) {
      if (event.key === 'Escape') {
        event.preventDefault();
        closeMenu();
      }
      return;
    }

    if (event.key === 'Escape') return;

    let handled = true;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case 'PageDown':
        goTo(current + 1);
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
      case 'PageUp':
        goTo(current - 1);
        break;
      case ' ':
        if (target instanceof HTMLElement && target.closest('button, a')) {
          handled = false;
          break;
        }
        goTo(current + 1);
        break;
      case 'Home':
        goTo(0);
        break;
      case 'End':
        goTo(sceneCount - 1);
        break;
      default:
        handled = false;
    }

    if (handled) event.preventDefault();
  }

  document.addEventListener('keydown', onKeydown);

  /* -----------------------------------------------------------------------
     Menu overlay
     ----------------------------------------------------------------------- */

  function getFocusable(container) {
    return Array.from(container.querySelectorAll(
      'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'
    )).filter((element) => element.offsetParent !== null);
  }

  function openMenu() {
    if (menuOpen) return;
    menuOpen = true;
    returnFocus = document.activeElement;
    document.body.classList.add('menu-open');
    document.body.classList.remove('has-scroll-cue');
    menuDialog.classList.add('is-open');
    menuDialog.setAttribute('aria-hidden', 'false');
    menuButton.setAttribute('aria-expanded', 'true');
    document.querySelector('.topbar').toggleAttribute('inert', true);
    main.toggleAttribute('inert', true);
    document.querySelector('.pager').toggleAttribute('inert', true);
    window.setTimeout(() => menuCloseButton.focus(), 30);
  }

  function closeMenu(options = {}) {
    const { restoreFocus = true } = options;
    if (!menuOpen) return;
    menuOpen = false;
    document.body.classList.remove('menu-open');
    menuDialog.classList.remove('is-open');
    menuDialog.setAttribute('aria-hidden', 'true');
    menuButton.setAttribute('aria-expanded', 'false');
    document.querySelector('.topbar').toggleAttribute('inert', false);
    main.toggleAttribute('inert', false);
    document.querySelector('.pager').toggleAttribute('inert', false);
    if (restoreFocus && returnFocus instanceof HTMLElement) {
      returnFocus.focus({ preventScroll: true });
    }
    window.requestAnimationFrame(updateScrollCue);
  }

  menuButton.addEventListener('click', openMenu);
  menuDialog.addEventListener('click', (event) => {
    if (event.target.closest('[data-close-menu]')) closeMenu();
  });

  menuDialog.addEventListener('keydown', (event) => {
    if (event.key !== 'Tab' || !menuOpen) return;
    const focusable = getFocusable(menuPanel);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  /* -----------------------------------------------------------------------
     Fullscreen
     ----------------------------------------------------------------------- */

  function syncFullscreenButton() {
    const active = Boolean(document.fullscreenElement);
    fullscreenButton.setAttribute('aria-label', active ? '退出全屏' : '进入全屏');
    fullscreenButton.title = active ? '退出全屏' : '全屏';
  }

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch (error) {
      // 浏览器可能因权限或环境拒绝全屏；保持页面状态即可。
    }
  }

  if (document.documentElement.requestFullscreen) {
    fullscreenButton.addEventListener('click', toggleFullscreen);
    document.addEventListener('fullscreenchange', syncFullscreenButton);
  } else {
    fullscreenButton.hidden = true;
  }

  /* -----------------------------------------------------------------------
     Boot
     ----------------------------------------------------------------------- */

  createTicks();
  initSceneState();
  syncFullscreenButton();
  wireScrollCue();
  window.requestAnimationFrame(updateScrollCue);

  scrollCue.addEventListener('click', () => {
    const inner = scenes[current].querySelector('.scene__inner');
    if (!inner) return;
    inner.scrollBy({
      top: Math.max(220, Math.round(inner.clientHeight * 0.72)),
      behavior: prefersReducedMotion.matches ? 'auto' : 'smooth',
    });
  });

  window.addEventListener('popstate', () => {
    const index = readHashIndex();
    if (index !== null) goTo(index, { push: false });
  });

  window.addEventListener('resize', () => {
    if (menuOpen && window.innerWidth > 1000) closeMenu({ restoreFocus: false });
    window.requestAnimationFrame(updateScrollCue);
  });

  // Start the first state cleanly if a deep link pointed below the cover.
  if (current !== 0) {
    // initSceneState already applied the deep-linked scene; no transition needed.
    syncInert();
  }

  // Expose a small debug surface for automated visual checks.
  window.__avmIntro = {
    goTo,
    get current() { return current; },
    get count() { return sceneCount; },
  };
})();
