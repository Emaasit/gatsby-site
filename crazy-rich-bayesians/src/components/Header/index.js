import React from 'react'
import { Link } from 'gatsby'
import * as config from '../../../config'
import mixpanel from 'mixpanel-browser';
import { MixpanelProvider, MixpanelConsumer } from 'react-mixpanel';

const Header = () => (
  <header className='bg-white black-80 tc pv4 avenir'>
    <h1 className='mt2 mb0 baskerville i fw1 f1'>{config.siteTitle}</h1>
    <h2 className='mt2 mb0 f6 fw5 ttu tracked'>Bayesian Modeling, Machine Learning, Behavior Analytics and Startups</h2>
    <nav className='bt bb tc mw8 center mt4'>
      <Link className='f6 f5-l fw5 link bg-animate black-80 hover-bg-lightest-blue dib pa3 ph4-l' to='/'>Home</Link>
      <Link className='f6 f5-l fw5 link bg-animate black-80 hover-bg-light-blue dib pa3 ph4-l' to='/about'>About</Link>
      <a className='f6 f5-l fw5 link bg-animate black-80 hover-bg-light-blue dib pa3 ph4-l' href='https://scholar.google.com/citations?user=kALpX2wAAAAJ&hl=en' target='_blank' rel='noopener nofollow noreferrer'>Publications</a>
      <a className='f6 f5-l fw5 link bg-animate black-80 hover-bg-light-blue dib pa3 ph4-l' href='https://www.dropbox.com/s/j9jobrzdknz6mma/CV_Daniel_Emaasit.pdf?dl=0' target='_blank' rel='noopener nofollow noreferrer'>CV</a>
      <a className='f6 f5-l fw5 link bg-animate black-80 hover-bg-light-blue dib pa3 ph4-l' href='https://github.com/Emaasit' target='_blank' rel='noopener nofollow noreferrer'>Software</a>
      <Link className='f6 f5-l fw5 link bg-animate black-80 hover-bg-light-blue dib pa3 ph4-l'
        to='/contact?no-cache=1'>Contact</Link>
      <Link className='f6 f5-l fw5 link bg-animate black-80 hover-bg-light-blue dib pa3 ph4-l' to='/search'>Search</Link>
    </nav>
    {/* <script type="text/javascript">
    mixpanel.track_links("#nav a", "click nav link", {
      "referrer": document.referrer
      });
    </script> */}
  </header>
)

export default Header
