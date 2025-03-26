import React from "react";

const Footer = () => {
  return (
    <footer className="bg-gray-100 text-gray-700 py-6 px-4">
      {/* Container to center content */}
      <div className="max-w-screen-xl mx-auto">
        {/* Top links row */}
        <div className="text-sm flex flex-wrap gap-2 mb-4">
          <a href="#" className="hover:underline">
            Privacy, Cookies, Security & Legal
          </a>
          <span>|</span>
          <a href="#" className="hover:underline">
            Terms of Use
          </a>
          <span>|</span>
          <a href="#" className="hover:underline">
            Online Access Agreement
          </a>
          <span>|</span>
          <a href="#" className="hover:underline">
            About Wells Fargo
          </a>
          {/* Add more links if needed */}
        </div>

        {/* Investment disclaimers (responsive text) */}
        <div className="text-xs md:text-sm space-y-2 mb-4 leading-snug">
          <p>
            <strong>Investment and Insurance Products are:</strong> Not Insured
            by the FDIC or Any Federal Government Agency • Not a Deposit or
            Other Obligation of, or Guaranteed by, the Bank or Any Bank
            Affiliate • Subject to Investment Risks, Including Possible Loss of
            the Principal Amount Invested
          </p>
          <p>
            Wells Fargo Advisors is a trade name used by Wells Fargo Clearing
            Services, LLC (WFCS) and Wells Fargo Advisors Financial Network,
            LLC, Members SIPC, separate registered broker-dealers and non-bank
            affiliates of Wells Fargo &amp; Company.
          </p>
          <p>
            Android, Google Play, and the Google Play logo are trademarks of
            Google LLC. Apple, the Apple logo, Apple Pay, Apple Watch, iBeacon,
            iPad, iPhone, Mac, Safari, and Touch ID are trademarks of Apple
            Inc., registered in the U.S. and other countries.
          </p>
        </div>

        {/* Bottom text rows */}
        <div className="text-xs md:text-sm space-y-1 mb-4">
          <p>© 2025 Wells Fargo Bank, N.A. Member FDIC. All rights reserved.</p>
          <p>Equal Housing Lender</p>
          <p>NMLSR ID 399801</p>
        </div>

        {/* Additional disclaimers / references if needed */}
        <div className="text-xs md:text-sm text-gray-500">
          <p>LRC-2024</p>
          <p>PW-202308-7770-1.0-1</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
