"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function Home() {
  const [generatedText, setGeneratedText] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    try {
      const response = await fetch(`http://0.0.0.0:8000/generate`)
      const data = await response.json()
      console.log(data)
      setGeneratedText(data.generated_text)
    } catch (error) {
      console.error("Error generating text:", error)
      setGeneratedText("An error occurred while generating text.")
    }
    setIsLoading(false)
  }

  return (
    <main className="container mx-auto p-4 flex items-center justify-center flex-col w-fit">
      <h1 className="text-4xl font-bold mb-8 text-center ">Bigram Language Model</h1>
      <Card className="w-fit">
        <CardHeader>
          <CardTitle>Generate Text</CardTitle>
          <CardDescription>
            Click the button below to generate text using a bigram language model
          </CardDescription>
        </CardHeader>
        <CardContent className="w-full">
          <form onSubmit={handleSubmit} className="">
         
            
            <Button type="submit" disabled={isLoading} className="w-full">
              {isLoading ? "Generating..." : "Generate Text"}
            </Button>
          </form>
        </CardContent>
      </Card>
      {generatedText && (
        <Card className="mt-8 w-full">
          <CardHeader>
            <CardTitle>Generated Text</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-wrap">{generatedText}</p>
          </CardContent>
        </Card>
      )}
    </main>
  )
}
