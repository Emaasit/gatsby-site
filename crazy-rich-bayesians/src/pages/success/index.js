import React from 'react'
import { Link } from 'gatsby'
import 'tachyons-sass/tachyons.scss'

const SuccessPage = () => (
  <section className='vh-100 avenir'>
    <header className='tc ph5 lh-copy'>
      <h1 className='f1 f-headline-l code mb3 fw9 dib tracked-tight light-pink'>Success!</h1>
      <h2 className='tc f1-l fw1'>Your Message Was Sent Successfully.</h2>
    </header>
    <p className='fw1 i tc mt4 mt5-l f4 f3-l'>I will contact you ASAP.</p>
    <ul className='list tc pl0 w-100 mt5'>
      <li className='dib'><Link className='f5 f4-ns link black db pv2 ph3 hover-light-blue' to='/' replace>Home</Link></li>
      <li className='dib'><Link className='f5 f4-ns link black db pv2 ph3 hover-light-blue' to='/about' replace>About</Link></li>
      <li className='dib'><Link className='f5 f4-ns link black db pv2 ph3 hover-light-blue' to='/contact' replace>Contact</Link>
      </li>
      <li className='dib'><Link className='f5 f4-ns link black db pv2 ph3 hover-light-blue' to='/tags' replace>Tags</Link></li>
    </ul>
  </section>
)

export default SuccessPage
