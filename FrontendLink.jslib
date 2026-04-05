mergeInto(LibraryManager.library, {
  NotifyFrontendLocationChanged: function (index) {
    if (typeof window.updateActiveLocation === 'function') {
      window.updateActiveLocation(index);
    }
  }
});