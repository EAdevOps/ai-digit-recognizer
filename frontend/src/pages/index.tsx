import Head from "next/head";
import DigitCanvas from "@/components/DigitCanvas";

export default function Home() {
  return (
    <>
      <Head>
        <title>AI Digit Recognizer</title>
        <meta name="viewport" content="initial-scale=1, width=device-width" />
      </Head>
      <main className="min-h-screen p-6">
        <div className="mx-auto max-w-3xl">
          <DigitCanvas />
        </div>
      </main>
    </>
  );
}
