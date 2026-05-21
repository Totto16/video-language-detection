import { bootstrapApplication, platformBrowser } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { App } from './app/app';

bootstrapApplication(App, appConfig, {
    platformRef: platformBrowser()
}).catch((err) => console.error(err));
